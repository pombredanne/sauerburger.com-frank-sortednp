#!/bin/bash

echo "$GPG_KEY" | gpg --import

pip install twine
twine upload --sign dist/* wheelhouse/*
