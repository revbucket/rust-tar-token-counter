use std::ffi::OsStr;
use std::hash::Hasher;
use std::hash::Hash;
use std::hash::DefaultHasher;
use std::io::Read;
use std::time::Instant;
use anyhow::{anyhow, bail, Result, Error};
use clap::Parser;
use std::path::{PathBuf};
use std::io::{BufReader, Cursor, Write};
use std::fs::{File};

use std::thread::available_parallelism;
use std::collections::{HashMap};
use std::sync::{Arc, Mutex};


use threadpool::ThreadPool;
use crate::s3::{is_s3, expand_s3_dir, get_reader_from_s3, write_cursor_to_s3};
use serde_json::{json};
use tar::{Archive};
use serde_json;
use serde_json::from_slice;
use glob::glob;
use dashmap::DashMap;
use indicatif::{ProgressBar,ProgressStyle};
use zstd::stream::read::Decoder as ZstdDecoder;

use rayon::prelude::*;

use flate2::read::MultiGzDecoder;


pub mod s3;


/*===================================================
=                    Utilities                      =
===================================================*/

#[derive(Parser, Debug)]
struct Args {
    /// (List of) directories/files (on s3 or local) that are jsonl.gz or jsonl.zstd files
    #[arg(required=true, long)]
    input: Vec<PathBuf>,


    /// Output location (may be an s3 uri)
    #[arg(required=true, long)]
    output: PathBuf,

    /// Number of threads to use 
    #[arg(long, default_value_t=0)]
    threads: usize,
}





pub(crate) fn expand_dirs(paths: Vec<PathBuf>, ext: Option<&str>) -> Result<Vec<PathBuf>> {
    // For local directories -> does a glob over each directory to get all files with given extension
    // For s3 directories -> does an aws s3 ls to search for files
    let ext = ext.unwrap_or(".jsonl.gz"); // Defaults to jsonl.gz, json.gz

    let mut files: Vec<PathBuf> = Vec::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();


    for path in paths {
        if is_s3(path.clone()) {
            // Use async_std to block until we scour the s3 directory for files
            runtime.block_on(async {
                let s3_paths = expand_s3_dir(&path, Some(ext)).await.unwrap();
                files.extend(s3_paths);                
            });                
        }
        else if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
            let mut num_hits = 0;
            //for entry in glob(&format!("{}/**/*.json*.gz", path_str))? {
            for entry in glob(&format!("{}/**/*{}", path_str, ext))? {

                files.push(entry?.to_path_buf());
                num_hits += 1;
            }
            if num_hits == 0 {
                bail!("No JSON Gz files found in '{}'", path_str);
            }
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}


fn read_pathbuf_to_mem(input_file: &PathBuf) -> Result<BufReader<Cursor<Vec<u8>>>, Error> {
    // Generic method to read local or s3 file into memory
    let reader = if is_s3(input_file) {
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        match rt.block_on(get_reader_from_s3(input_file, Some(5))) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Error! {:?}", err);
                return Err(err.into());
            }
        }
    } else {
        let contents = read_local_file_into_memory(input_file).expect("Failed to read contents into memory");
        BufReader::new(contents)
    };
    Ok(reader)
} 


fn read_local_file_into_memory(input_file: &PathBuf) ->Result<Cursor<Vec<u8>>, Error>{
    // Takes a local file (must be local!) and reads it into a Cursor of bytes
    let mut file = File::open(input_file).expect("Failed to open file");

    let mut contents = Vec::new();
    let ext = input_file.extension().unwrap().to_string_lossy().to_lowercase();
    if ext == "gz" {
        // Gzip case        
        let mut decoder = MultiGzDecoder::new(file);
        decoder.read_to_end(&mut contents).expect("Failed to read loca gzip file");
    } else if ext == "zstd" || ext == "zst" {
        // Zstd case
        let mut decoder = ZstdDecoder::new(file).unwrap();
        decoder.read_to_end(&mut contents).expect("Failed to read local zstd file");
    } else {
        file.read_to_end(&mut contents).expect("Failed to read local file");

        // No compression case 
    }
    Ok(Cursor::new(contents))
}


fn hash_str(text: &OsStr, seed: usize) -> u64 {
    // Hashes a vector of type T into a u64 hash value
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    text.hash(&mut hasher);
    hasher.finish()
}



/*==============================================================
=                             Meat                             =
==============================================================*/




fn count_from_compressed_data(compressed_data: Vec<u8>, mut local_counter: HashMap<u64, usize>) -> Result<HashMap<u64, usize>, Error> {
    let mut decoder = MultiGzDecoder::new(&compressed_data[..]);
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();

    let numbers: Vec<u64> = from_slice(&decompressed_data).unwrap();
    for number in numbers {
        *local_counter.entry(number).or_insert(0) += 1;
    }
    Ok(local_counter)
}



async fn process_file(input: &PathBuf, global_counter: &Arc<DashMap<u64, usize>>) -> Result<(), Error> {
    // Count number of tokens in each context in each input 


    let reader = if is_s3(input) {
        get_reader_from_s3(input, Some(5)).await.unwrap()
    } else {
        BufReader::new(read_local_file_into_memory(&input).unwrap())
    };
    let mut tar = Archive::new(reader);
    let mut local_counter : HashMap<u64, usize> = HashMap::new(); 


    for entry in tar.entries()? {
        // iterate over entries and increment local hashmap
        let entry = entry?;
        let path = entry.path().unwrap();
        let path = path.as_os_str();

        let path_hash = hash_str(path, 1234);

        *local_counter.entry(path_hash).or_insert(0) += 1;
    }

    // plug these hash values into the global counter
    for (key, value) in local_counter.iter() {
        global_counter.entry(*key).or_insert(0);
        global_counter.alter(&key, |_, cur| {
            cur + value
        });
    }

    Ok(())

}
                      



fn main() -> Result<()> {
    let start_time = Instant::now();    
    let args = Args::parse();
    let threads = if args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        args.threads
    };    
    let input_files =  expand_dirs(args.input, Some(".tar")).unwrap() ;



    let pbar = ProgressBar::new(input_files.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    let token_counter = Arc::new(DashMap::new());


    // Step 2: Iterate over all files and process
    let threadpool = ThreadPool::new(threads);
    for input in input_files {    
        let pbar = pbar.clone();
        let token_counter = Arc::clone(&token_counter);
        threadpool.execute(move || {        
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
            let result = rt.block_on({
                let subresult = process_file(
                    &input,
                    &token_counter,
                    );              
                pbar.lock().unwrap().inc(1);
                subresult}
                );            
            match result {            
                Err(err) => {
                    eprintln!("Error processing {:?}; {:?}", input, err);
                },
                _ => {},
            };
        });
    }
    threadpool.join();
    

    // Step 3: finalize the dashmap into something we can save
    //token_counter.into_inner();
    println!("DASHMAP HAS LEN {:?}", token_counter.len());
    let token_counter : HashMap<u64, usize> = (<DashMap<u64, usize> as Clone>::clone(&token_counter))
        .into_iter()
        .par_bridge()
        .filter(|entry| entry.1 > 1)
        .map(|entry| (entry.0, entry.1))
        .collect();
    println!("TOKEN COUNTER HAS OUTPUT LEN {:?}", token_counter.len());

    let json_data = json!(token_counter);
    println!("DONE WITH JSONING, NOW UPLOADING");
    if is_s3(&args.output) {
        let json_bytes: Vec<u8> = serde_json::to_vec(&json_data).unwrap();
        let cursor = Cursor::new(json_bytes);
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();           
        rt.block_on(write_cursor_to_s3(&args.output, cursor)).unwrap();        
    } else {
        let mut file = File::create(args.output).unwrap();
        file.write_all(json_data.to_string().as_bytes()).unwrap();
    }

    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());

    Ok(())
}
