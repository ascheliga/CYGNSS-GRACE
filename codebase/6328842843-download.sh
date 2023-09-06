#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (ascheliga): " username
    username=${username:-ascheliga}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230401-S000000-E235959.04.V07A.HDF5"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230401-S000000-E235959.04.V07A.HDF5 -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230401-S000000-E235959.04.V07A.HDF5 | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o $stripped_query_params -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document $stripped_query_params --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230401-S000000-E235959.04.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230301-S000000-E235959.03.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230201-S000000-E235959.02.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2023/3B-MO.MS.MRG.3IMERG.20230101-S000000-E235959.01.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20221201-S000000-E235959.12.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20221101-S000000-E235959.11.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20221001-S000000-E235959.10.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220901-S000000-E235959.09.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220801-S000000-E235959.08.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220701-S000000-E235959.07.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220601-S000000-E235959.06.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220501-S000000-E235959.05.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220401-S000000-E235959.04.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220301-S000000-E235959.03.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220201-S000000-E235959.02.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2022/3B-MO.MS.MRG.3IMERG.20220101-S000000-E235959.01.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20211201-S000000-E235959.12.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20211101-S000000-E235959.11.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20211001-S000000-E235959.10.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210901-S000000-E235959.09.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210801-S000000-E235959.08.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210701-S000000-E235959.07.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210601-S000000-E235959.06.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210501-S000000-E235959.05.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210401-S000000-E235959.04.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210301-S000000-E235959.03.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210201-S000000-E235959.02.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2021/3B-MO.MS.MRG.3IMERG.20210101-S000000-E235959.01.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20201201-S000000-E235959.12.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20201101-S000000-E235959.11.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20201001-S000000-E235959.10.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200901-S000000-E235959.09.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200801-S000000-E235959.08.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200701-S000000-E235959.07.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200601-S000000-E235959.06.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200501-S000000-E235959.05.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200401-S000000-E235959.04.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200301-S000000-E235959.03.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200201-S000000-E235959.02.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2020/3B-MO.MS.MRG.3IMERG.20200101-S000000-E235959.01.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20191201-S000000-E235959.12.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20191101-S000000-E235959.11.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20191001-S000000-E235959.10.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190901-S000000-E235959.09.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190801-S000000-E235959.08.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190701-S000000-E235959.07.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190601-S000000-E235959.06.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190501-S000000-E235959.05.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190401-S000000-E235959.04.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190301-S000000-E235959.03.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190201-S000000-E235959.02.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2019/3B-MO.MS.MRG.3IMERG.20190101-S000000-E235959.01.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20181201-S000000-E235959.12.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20181101-S000000-E235959.11.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20181001-S000000-E235959.10.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180901-S000000-E235959.09.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180801-S000000-E235959.08.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180701-S000000-E235959.07.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180601-S000000-E235959.06.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180501-S000000-E235959.05.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180401-S000000-E235959.04.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180301-S000000-E235959.03.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180201-S000000-E235959.02.V07A.HDF5
https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/2018/3B-MO.MS.MRG.3IMERG.20180101-S000000-E235959.01.V07A.HDF5
EDSCEOF