# Purpose: Read the original file and create new shortened file

# Location of USGS shell script
sh_fp = "/global/scratch/users/ann_scheliga/aux_dam_datasets/Landsat8/"
sh_fn = "0373424596-download.sh"

# Grab original content and make edits
with open(sh_fp + sh_fn) as f:
    # Create a list of file contents (instead of a string)
    full_file = f.readlines()
    # Split content into sections
    front_matter = full_file[:97]
    urls = full_file[97:-1]
    back_matter = full_file[-1:]
    # Set filters for urls
    filter_strs = ["B03", "B05", "Fmask"]
    # Select only urls that contain any filter value
    urls_shortened = [url for url in urls if any(band in url for band in filter_strs)]
    print(len(urls_shortened))
    final_file = front_matter + urls_shortened + back_matter

# # Write the editted script to a new file.
with open(sh_fp + "download_script_B03_B05.sh", "w") as f:
    f.writelines(final_file)
