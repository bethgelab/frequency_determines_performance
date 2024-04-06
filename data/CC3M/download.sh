# Download the train and validation tsv from https://ai.google.com/research/ConceptualCaptions/download
# Then run this script
for i in {00000..00331}; do wget https://huggingface.co/datasets/zatepyakin/cc3m_min256_max512/resolve/main/$i.tar; done
