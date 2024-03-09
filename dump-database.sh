#!/bin/bash

# Set the -e option so that if any command fails the script will exit immediately
set -e

# Dump the sql database to a csv file
echo "Dumping the sql database to a csv file..."
mysql -u root -p$1 -e "USE galaxypedia; SELECT page_namespace, page_title \"page_name\", old_text \"content\" FROM page INNER JOIN slots on page_latest = slot_revision_id INNER JOIN slot_roles on slot_role_id = role_id AND role_name = 'main' INNER JOIN content on slot_content_id = content_id INNER JOIN text on substring( content_address, 4 ) = old_id AND left( content_address, 3 ) = \"tt:\" WHERE (page.page_namespace = 0 OR page.page_namespace = 4) AND page.page_is_redirect = 0 into outfile '/tmp/galaxypedia.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n';"

# Move the csv file to the current directory
echo "Moving the csv file to the current directory..."
sudo mv -f /tmp/galaxypedia.csv ./galaxypedia.csv.temp

# Change the owner of the file to ubuntu
echo "Changing the owner of the file to ubuntu..."
sudo chown $(whoami):$(whoami) galaxypedia.csv.temp

# Add the header to the csv file
echo "Adding the header to the csv file..."
(echo "page_namespace, page_title,content"; cat galaxypedia.csv.temp) > galaxypedia-$(date '+%Y-%m-%d').csv

# Remove the temporary file
echo "Removing the temporary file..."
rm galaxypedia.csv.temp
