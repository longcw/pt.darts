import os
import subprocess
import zipfile


def download_data():
    data_diar = "/data"
    aws_endpoint = "https://store-028.blobstore.apple.com"
    aws_region = "store-028"
    obj = "s3://display/longc/keyboard_aoi/keyboard_aoi_data.zip"

    dest = data_diar
    print("Downloading {} to {}".format(obj, dest))
    if not os.path.isdir(dest):
        os.makedirs(dest)
    command = "aws --endpoint-url {} s3 cp {} {} --region {}".format(
        aws_endpoint, obj, dest, aws_region
    )
    subprocess.check_call(command.split())
    print("Downloading Finished.")

    print("Unzipping to {}".format(dest))
    with zipfile.ZipFile(os.path.join(dest, os.path.basename(obj)), "r") as f:
        f.extractall(dest)
    print("Unzipping Finished.")


if __name__ == "__main__":
    download_data()
