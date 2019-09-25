#!/bin/bash

for file in splits/*; do
	python get_vocab.py < $file > $file.vocab &
done
