#!/usr/bin/env bash
set -e

BASE="headers"
mkdir -p "$BASE"

cd "$BASE"

echo "=== Downloading Enron Maildir (working mirror) ==="
wget -c https://www.cs.cmu.edu/~./enron/enron_mail_20110402.tgz -O enron_mail.tgz
mkdir -p enron
mkdir -p enron_20110402
tar -xzf enron_mail.tgz -C enron_20110402 --strip-components=1
mv enron_20110402/maildir/* ./enron
rm -rf enron_20110402
rm -rf enron_mail.tgz

echo "=== Downloading SpamAssassin Public Corpus ==="
mkdir -p spamassassin
cd spamassassin

wget -c https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2
wget -c https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2

for f in *.tar.bz2; do
  echo "Extracting $f"
  tar -xjf "$f"
  rm -f "$f"
done

echo "=== Done downloading & extracting corpora ==="
