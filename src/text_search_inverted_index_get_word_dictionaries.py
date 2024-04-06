import argparse, spacy, os, gc, numpy, pandas, pickle, re

def spacy_parse(documents, batch_size):
    """
    Parse a list of documents using the Spacy LG model.

    Args:
        documents (list): A list of documents to be parsed.
        batch_size (int): The batch size for processing the documents.

    Returns:
        list: A list of processed documents.

    """
    # Run this on CPU
    parser = spacy.load("en_core_web_lg") # Accurate but far slower: en_core_web_trf
    print(f'==> Finished loading Spacy LG model!')
    proc_docs = list(parser.pipe(documents, batch_size=batch_size, n_process=1))
    return proc_docs

def save_file(content, filename):
    with open(f'{filename}.pkl', "wb") as output_file:
        pickle.dump(content, output_file)


def load_file(filename):
    with open(f'{filename}.pkl', "rb") as output_file:
        content = pickle.load(output_file)
    return content


def read_files(args):
    """
    Read files based on the given arguments and return the links and texts.

    Args:
        args (argparse.Namespace): The arguments passed to the function.

    Returns:
        tuple: A tuple containing two lists - links and texts.

    Raises:
        AssertionError: If the length of new_texts is not equal to the length of texts.

    """
    if args.dataset == 'CC3M' and args.rewrite == 'None':
        df = pandas.read_csv(args.path, sep='\t', names=['Text', 'Link'])
        links = df['Link'].to_list()
        texts = df['Text'].to_list()
    elif args.dataset == 'CC12M' and args.rewrite == 'None':
        df = pandas.read_csv(args.path, sep='\t', names=['Link', 'Text'])
        links = df['Link'].to_list()
        texts = df['Text'].to_list()
    elif args.dataset == 'LAION400M':
        df = pandas.read_parquet(args.path)
        df['caption'].replace('', numpy.nan, inplace=True)
        df['url'].replace('', numpy.nan, inplace=True)
        df.dropna(subset=['caption','url'], inplace=True)
        links = df['url'].to_list()
        texts = df['caption'].to_list()
    elif args.dataset == 'LAION2B':
        df = pandas.read_parquet(args.path)
        df['TEXT'].replace('', numpy.nan, inplace=True)
        df['URL'].replace('', numpy.nan, inplace=True)
        df.dropna(subset=['URL','TEXT'], inplace=True)
        links = df['URL'].to_list()
        texts = df['TEXT'].to_list()
    elif args.dataset == 'YFCC15M':
        df = pandas.read_csv(args.path, sep='\t', names=['Link', 'Text'])
        links = df['Link'].to_list()
        texts = df['Text'].to_list()
    elif (args.dataset == 'CC3M' or args.dataset == 'CC12M') and args.rewrite != 'None':
        df = pandas.read_csv(args.path, names=['Image','Text', 'Link'])
        links = df['Link'].to_list()
        texts = df['Text'].to_list()
        new_texts = []
        with open(args.rewrite_path, 'r') as f:
            for line in f:
                new_texts.append(line.strip())
        assert(len(new_texts)==len(texts))
        texts = new_texts

    links_per_chunk = (len(links)//args.num_chunks)+1
    links, texts = links[args.chunk_idx*links_per_chunk:(args.chunk_idx+1)*links_per_chunk], texts[args.chunk_idx*links_per_chunk:(args.chunk_idx+1)*links_per_chunk]
    return links, texts

### DICTIONARIES ###
# Structure of the dictionaries
# Key: word
# Value: List[int]: Each integer is a line number (global): chunk_idx*links_per_chunk + line_number_in_chunk
# # Frequency of word: len(my_dictionary[key])

def create_unigram_dictionary_spacy(proc_docs):
    unigram_dictionary = {}
    for i in range(len(proc_docs)):
        # Get all nouns in the document
        nouns = list(set([re.sub(r'[^A-Za-z ]', '', token.lemma_) for token in proc_docs[i] if token.pos_ in ['NOUN','PROPN']]))
        # Add nouns to dictionary
        for noun in nouns:
            if noun not in unigram_dictionary:
                unigram_dictionary[noun] = [i]
            unigram_dictionary[noun].append(i)

    return unigram_dictionary


def create_extended_dictionary_spacy(proc_docs):
    extended_dictionary = {}

    for i in range(len(proc_docs)):
        seen_chunks = []
        for chunk in proc_docs[i].noun_chunks:
            if re.sub(r'[^A-Za-z ]', '', chunk.text) in seen_chunks:
                continue
            if chunk not in extended_dictionary:
                extended_dictionary[re.sub(r'[^A-Za-z ]', '', chunk.text)] = [i]
            else:
                extended_dictionary[re.sub(r'[^A-Za-z ]', '', chunk.text)].append(i)
            seen_chunks.append(re.sub(r'[^A-Za-z ]', '', chunk.text))
        
    return extended_dictionary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['CC3M','CC12M','LAION400M','LAION2B','YFCC15M'])
    parser.add_argument("--path", type=str)
    parser.add_argument("--rewrite", type=str, default='None', choices=['None','ChatGPT','Bard','COCO','Human'])
    parser.add_argument("--rewrite_path", type=str)
    parser.add_argument("--save_filepath", type=str, default='test')
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    assert(args.chunk_idx<args.num_chunks)
    print(f'==> Args for this experiment are: {args}')

    # Get Data
    links, texts = read_files(args=args)
    assert(len(links)==len(texts)), 'Lengths of links and texts different!'
    print(f'==> Finished Reading files! Length: {len(links)}')
    
    # Process Data
    proc_docs = spacy_parse(documents=texts, batch_size=args.batch_size)
    print(f'==> Finished parsing files! Length: {len(proc_docs)}')
    save_file(content=links, filename=args.save_filepath+'_'+str(args.chunk_idx)+'_links')
    save_file(content=texts, filename=args.save_filepath+'_'+str(args.chunk_idx)+'_captions')
    assert(len(proc_docs)==len(links))

    # Save RAM Space
    del links, texts
    gc.collect()

    # Adding unigram and extended dictionary creation scripts
    if not os.path.exists(args.save_filepath+'_'+str(args.chunk_idx)+'_unigram_dict.pkl'):
        print(f'==> Getting unigram dictionaries..')
        unigram_dictionary = create_unigram_dictionary_spacy(proc_docs=proc_docs)
        save_file(content=unigram_dictionary, filename=args.save_filepath+'_'+str(args.chunk_idx)+'_unigram_dict')
        print(f'==> Unigram dictionaries created and saved..')
        del unigram_dictionary
        gc.collect()

    if not os.path.exists(args.save_filepath+'_'+str(args.chunk_idx)+'_extended_dict.pkl'):
        print(f'==> Getting multigram dictionaries..')
        extended_dictionary = create_extended_dictionary_spacy(proc_docs=proc_docs)
        save_file(content=extended_dictionary, filename=args.save_filepath+'_'+str(args.chunk_idx)+'_extended_dict')
        print(f'==> Multigram dictionaries created and saved..')
        del extended_dictionary
        gc.collect()