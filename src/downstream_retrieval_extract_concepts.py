import argparse, spacy, os, gc, numpy, pandas, pickle, re
from datasets import load_dataset
from tqdm import tqdm

def data_load(args):
    if args.dataset == 'coco':
        dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")
    else:
        dataset = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval")
    data_split = dataset['test']

    caps = []
    for d in data_split['caption']:
        caps += d[:5]
    return caps

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

def extract_concepts(proc_docs):
    concepts = {}
    concepts_to_text_samples_map = {}
    text_samples_to_concepts_map = {}
    concepts_to_image_samples_map = {}
    image_samples_to_concepts_map = {}

    for i in tqdm(range(len(proc_docs)), ascii=True, total=len(proc_docs)):
        # Get all nouns in the document
        nouns = list(set([re.sub(r'[^A-Za-z ]', '', token.lemma_).lower() for token in proc_docs[i] if token.pos_ in ['NOUN','PROPN']]))
        text_samples_to_concepts_map[i] = []

        # Add nouns
        for noun in nouns:
            concepts[noun] = 1
            if noun not in concepts_to_text_samples_map:
                concepts_to_text_samples_map[noun] = []
            concepts_to_text_samples_map[noun].append(i)
            text_samples_to_concepts_map[i].append(noun)

    # get image concepts
    for index in range( len(text_samples_to_concepts_map) // 5 ):
        matched_samps = [index * 5, (index + 1) * 5]
        concs = []
        for ind in range(matched_samps[0], matched_samps[1]):
            concs += text_samples_to_concepts_map[ind]
        concs = list(set(concs))
        image_samples_to_concepts_map[index] = concs
        for con in concs:
            if con not in concepts_to_image_samples_map:
                concepts_to_image_samples_map[con] = []
            concepts_to_image_samples_map[con].append(index)

    filtered_concepts = []
    for k,v in concepts_to_image_samples_map.items():
        if len(v) >= 10:
            filtered_concepts.append(k)

    filtered_concepts_to_text_samples_map = {}
    filtered_text_samples_to_concepts_map = {}
    filtered_concepts_to_image_samples_map = {}
    filtered_image_samples_to_concepts_map = {}

    for concept in concepts_to_text_samples_map:
        if concept in filtered_concepts:
            filtered_concepts_to_text_samples_map[concept] = concepts_to_text_samples_map[concept]

    for concept in concepts_to_image_samples_map:
        if concept in filtered_concepts:
            filtered_concepts_to_image_samples_map[concept] = concepts_to_image_samples_map[concept]

    for sample in text_samples_to_concepts_map:
        new_arr = []
        for con in text_samples_to_concepts_map[sample]:
            if con in filtered_concepts:
                new_arr.append(con)
        filtered_text_samples_to_concepts_map[sample] = new_arr

    for sample in image_samples_to_concepts_map:
        new_arr = []
        for con in image_samples_to_concepts_map[sample]:
            if con in filtered_concepts:
                new_arr.append(con)
        filtered_image_samples_to_concepts_map[sample] = new_arr

    return filtered_concepts, filtered_concepts_to_text_samples_map, filtered_text_samples_to_concepts_map, filtered_concepts_to_image_samples_map, filtered_image_samples_to_concepts_map

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['coco', 'flickr'], default='coco')
    parser.add_argument("--features_path", type=str, default='./features')
    parser.add_argument("--batch_size", type=int, default=5000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(f'==> Args for this experiment are: {args}')

    # Get Dataset
    text_captions = data_load(args)
    print(f'==> Finished loading captions! Length: {len(text_captions)}')
    
    # Process Data
    proc_docs = spacy_parse(documents=text_captions, batch_size=args.batch_size)
    print(f'==> Finished parsing files! Length: {len(proc_docs)}')
    print(len(proc_docs))

    # Save RAM Space
    del text_captions
    gc.collect()

    # Adding unigram and extended dictionary creation scripts
    print(f'==> Getting unigram dictionaries..')
    concept_list, concepts_to_text_samples_map, text_samples_to_concepts_map, concepts_to_image_samples_map, image_samples_to_concepts_map = extract_concepts(proc_docs=proc_docs)

    assert len(concepts_to_text_samples_map) == len(concepts_to_image_samples_map)
    # ensure that there are 5 text captions per image
    assert len(text_samples_to_concepts_map) == 5 * len(image_samples_to_concepts_map)

    print('Num concepts: {}'.format(len(concept_list)))
    with open(os.path.join(args.features_path, '{}_concept_list.pkl'.format(args.dataset)), 'wb') as f:
        pickle.dump(concept_list, f)
    with open(os.path.join(args.features_path, '{}_concepts_to_text_samples_map.pkl'.format(args.dataset)), 'wb') as f:
        pickle.dump(concepts_to_text_samples_map, f)
    with open(os.path.join(args.features_path, '{}_text_samples_to_concepts_map.pkl'.format(args.dataset)), 'wb') as f:
        pickle.dump(text_samples_to_concepts_map, f)
    with open(os.path.join(args.features_path, '{}_concepts_to_image_samples_map.pkl'.format(args.dataset)), 'wb') as f:
        pickle.dump(concepts_to_image_samples_map, f)
    with open(os.path.join(args.features_path, '{}_image_samples_to_concepts_map.pkl'.format(args.dataset)), 'wb') as f:
        pickle.dump(image_samples_to_concepts_map, f)