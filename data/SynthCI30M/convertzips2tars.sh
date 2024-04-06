DIR="./images"

cd "$DIR"

success_count=0

for zipfile in *.zip; do
  base=${zipfile%.zip}

  # Convert zip to tar using zip2tar
  zip2tar "$zipfile"

  # Check if tar file was created successfully
  if [ -f "${base}.tar" ]; then
    ((success_count++))
    # Delete the original zip file
    rm "$zipfile"
    echo "Conversion succesful for $zipfile"
    echo "$success_count"
  else
    echo "Conversion failed for $zipfile"
  fi
done

echo "$success_count zip files successfully converted to tar."
