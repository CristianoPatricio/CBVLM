# Download DDI - Diverse Dermatology Images

## 1. Register and get download link from the official website

DDI - Diverse Dermatology Images Website: https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965

Link: https://aimistanforddatasets01.blob.core.windows.net/ddidiversedermatologyimages?sv=2019-02-02&sr=c&sig=sPHtATha4v9jdnVXh7h4lyOGPqbZFCexDwatqF%2B7w%2FM%3D&st=2024-04-29T10%3A54%3A40Z&se=2024-05-29T10%3A59%3A40Z&sp=rl

## 2. Install `azcopy` on Linux
```bash
# 1. Downloads the zip setup file on our local machine
wget https://aka.ms/downloadazcopy-v10-linux 

# 2. Extracts the file content from the archive file downloaded in step 1
tar -xvf downloadazcopy-v10-linux

# 3. To be executed if we had a previous version of Azcopy in our machine. This command removes the bin file for the previous installation
sudo rm /usr/bin/azcopy 

# 4. This command moves the azcopy files in the bin folder of the user
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
```

## 3. Downloading data from Azure server to our local machine
```bash
sudo azcopy copy --recursive "https://aimistanforddatasets01.blob.core.windows.net/ddidiversedermatologyimages?sv=2019-02-02&sr=c&sig=sPHtATha4v9jdnVXh7h4lyOGPqbZFCexDwatqF%2B7w%2FM%3D&st=2024-04-29T10%3A54%3A40Z&se=2024-05-29T10%3A59%3A40Z&sp=rl" "/path/to/save/data"
```