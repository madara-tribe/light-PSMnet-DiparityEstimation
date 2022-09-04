#/bin/sh
rm -r logs checkpoint depth
rm pic/*
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name '.DS_Store' -type f -ls -delete
