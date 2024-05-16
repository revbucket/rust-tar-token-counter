use serde_json::Value;
use std::io::BufRead;
use std::time::Instant;
use anyhow::{anyhow, bail, Result, Error};
use clap::Parser;
use std::path::{PathBuf};
use std::io::{BufReader, Cursor, Read, Write};
use std::fs::{File};

use std::thread::available_parallelism;

use std::sync::{Arc, Mutex};


use threadpool::ThreadPool;
use crate::s3::{is_s3, expand_s3_dir, get_reader_from_s3, write_cursor_to_s3};


use serde_json;

use glob::glob;

use indicatif::{ProgressBar,ProgressStyle};
use zstd::stream::read::Decoder as ZstdDecoder;

use base64::{engine::general_purpose, Engine as _};
use tiktoken_rs::CoreBPE;
use rustc_hash::FxHashMap;



use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;


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


    /// Extension of file 
    #[arg(required=true, long)]
    ext: String,
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


fn compress_gzip(data: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}


/*==============================================================
=                             Meat                             =
==============================================================*/

fn load_tiktoken_tokenizer() -> Result<CoreBPE> {
    // Loats the tiktoken tokenizer. Some magic strings here, but don't worry about it ;)
    let pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let tiktoken_data = include_str!("../EleutherAI_gpt-neox-20b.tiktoken");

    let mut encoder = FxHashMap::default();

    for line in tiktoken_data.lines() {
        let mut parts = line.split(' ');
        let raw = parts.next().unwrap();
        let token = &general_purpose::STANDARD.decode(raw)?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);        
    }
    let special_tokens = FxHashMap::default();
    let bpe = CoreBPE::new(
        encoder, 
        special_tokens,
        pattern,
        )?;

    Ok(bpe)
}





async fn process_file(input: &PathBuf, global_toklengths: &Arc<Mutex<Vec<usize>>>, global_bytelengths: &Arc<Mutex<Vec<usize>>>,
                      pbar: &Arc<Mutex<ProgressBar>>) -> Result<(), Error> {
    // Count document lengths (in characters, and tokens) and return them

    let reader = if is_s3(input) {
        get_reader_from_s3(input, Some(5)).await.unwrap()
    } else {
        BufReader::new(read_local_file_into_memory(&input).unwrap())
    };
    let tokenizer = load_tiktoken_tokenizer()?;

    let mut local_bytelengths: Vec<usize> = Vec::new();
    let mut local_toklengths: Vec<usize> = Vec::new();


    for line in reader.lines() {
        let line = line?; 
        let json: Value = serde_json::from_str(&line)?;
        let text = json["text"].as_str().unwrap();         
        local_bytelengths.push(text.len());
        local_toklengths.push(tokenizer.encode_with_special_tokens(text).len());
    }


    global_bytelengths.lock().unwrap().extend(local_bytelengths);
    global_toklengths.lock().unwrap().extend(local_toklengths);


    pbar.lock().unwrap().inc(1);
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
    let input_files =  expand_dirs(args.input, Some(&args.ext)).unwrap() ;



    let pbar = ProgressBar::new(input_files.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));




    // Step 2: Iterate over all files and process
    let global_toklengths: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let global_bytelengths: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

    let threadpool = ThreadPool::new(threads);


    for input in input_files {    
        let pbar = pbar.clone();
        let global_toklengths = Arc::clone(&global_toklengths);
        let global_bytelengths = Arc::clone(&global_bytelengths);
        threadpool.execute(move || {        
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
            let result = rt.block_on({
                let subresult = process_file(
                    &input,
                    &global_toklengths,
                    &global_bytelengths,
                    &pbar,
                    );              
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

    let json_data = serde_json::json!({
        "bytelengths": global_bytelengths.lock().unwrap().clone(),
        "toklengths": global_toklengths.lock().unwrap().clone(),
    });
    let json_bytes: Vec<u8> = serde_json::to_vec(&json_data).unwrap();
    let gzip_bytes: Vec<u8> = compress_gzip(&json_bytes);    

    if is_s3(&args.output) {
        let cursor = Cursor::new(gzip_bytes);
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();           
        rt.block_on(write_cursor_to_s3(&args.output, cursor)).unwrap();        
    } else {
        let mut file = File::create(args.output).unwrap();
        file.write_all(gzip_bytes.as_slice()).unwrap();
    }

    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());

    Ok(())
}
