# Refer to the link for more details--https://huggingface.co/datasets/hammh0a/SynthCLIP
# wget -c https://huggingface.co/datasets/hammh0a/SynthCLIP/resolve/main/SynthCI-30/combined_images_and_captions.csv

target_directory="./images"

# Create the directory if it doesn't exist
mkdir -p "$target_directory"

# Change into the target directory
cd "$target_directory"

for i in {0..3038}; do wget -c https://huggingface.co/datasets/hammh0a/SynthCLIP/resolve/main/SynthCI-30/data/$i.zip; done
# post downloading, run convertips2tars.sh to convert the zips to tar files