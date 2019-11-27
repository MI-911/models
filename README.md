# Models
Learning models and data processing libraries.

## Download MindReader dataset
Run the `download_mindreader` function in `./data/download_data.py`. By default, the `ratings.csv` and `entities.csv` files will be saved in `./data/mindreader/`. Make sure this directory exists beforehand.
When the files have been downloaded, run the `preprocess_user_major` function in the same file to get the ratings in a user-major fashion (i.e. a dictionary of user ids and their corresponding ratings separated as movie and entity ratings).