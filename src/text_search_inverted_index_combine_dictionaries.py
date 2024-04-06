import argparse, pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CC', choices=['CC','LAION'])
    parser.add_argument("--save_filepath", type=str, default='test')
    parser.add_argument("--total_chunks", type=int, default=1)
    args = parser.parse_args()
    return args

def save_file(content, filename):
    with open(f'{filename}.pkl', "wb") as output_file:
        pickle.dump(content, output_file)

def load_file(filename):
    with open(f'{filename}.pkl', "rb") as output_file:
        content = pickle.load(output_file)
    return content


if __name__ == '__main__':
    args = get_args()
    print(f'==> Args for this experiment are: {args}')
    combined_links = []
    offset = 0
    combined_unigram_dictionary = {}

    for idx in range(args.total_chunks):
        print(f'==> Loading dictionary {idx}')


        if args.dataset == 'CC':
            dictionary = load_file(filename=args.save_filepath+'_'+str(idx)+'_unigram_dict')
            links = load_file(filename=args.save_filepath+'_'+str(idx)+'_links')

        elif args.dataset == 'LAION':
            dictionary = load_file(filename=args.save_filepath+'/laion400m_'+str(idx)+'_spacy_0_unigram_dict')
            links = load_file(filename=args.save_filepath+'/laion400m_'+str(idx)+'_spacy_0_links')

        

        # Combine unigram dictionaries
        for key in dictionary:
            idxes = dictionary[key]
            idxes = [idx+offset for idx in idxes]
            if key not in combined_unigram_dictionary:
                combined_unigram_dictionary[key] = idxes
            else:
                combined_unigram_dictionary[key] += idxes
        
        offset += len(links)
        combined_links += links

    if args.dataset == 'CC':
        save_file(content=combined_unigram_dictionary, filename=args.save_filepath+'_combined_unigram_dict')
        save_file(content=combined_links, filename=args.save_filepath+'_combined_links')
    elif args.dataset == 'LAION':
        save_file(content=combined_unigram_dictionary, filename=args.save_filepath+'/laion400m_spacy_combined_unigram_dict')
        save_file(content=combined_links, filename=args.save_filepath+'/laion400m_spacy_combined_links')

    offset = 0
    combined_captions = []
    combined_extended_dictionary = {}
    for idx in range(args.total_chunks):
        print(f'==> Loading dictionary {idx}')


        if args.dataset == 'CC':
            extended_dictionary = load_file(filename=args.save_filepath+'_'+str(idx)+'_extended_dict')
            captions = load_file(filename=args.save_filepath+'_'+str(idx)+'_captions')

        elif args.dataset == 'LAION':
            extended_dictionary = load_file(filename=args.save_filepath+'/laion400m_'+str(idx)+'_spacy_0_extended_dict')
            captions = load_file(filename=args.save_filepath+'/laion400m_'+str(idx)+'_spacy_0_captions')

        # Combine extended dictionaries
        for key in extended_dictionary:
            idxes = extended_dictionary[key]
            idxes = [idx+offset for idx in idxes]
            if key not in combined_extended_dictionary:
                combined_extended_dictionary[key] = idxes
            else:
                combined_extended_dictionary[key] += idxes
        
        offset += len(captions)
        combined_captions += captions

    if args.dataset == 'CC':
        save_file(content=combined_extended_dictionary, filename=args.save_filepath+'_combined_extended_dict')
        save_file(content=combined_captions, filename=args.save_filepath+'_combined_captions')
    elif args.dataset == 'LAION':
        save_file(content=combined_extended_dictionary, filename=args.save_filepath+'/laion400m_spacy_combined_extended_dict')
        save_file(content=combined_captions, filename=args.save_filepath+'/laion400m_spacy_combined_captions')
