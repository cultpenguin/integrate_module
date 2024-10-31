find ./ -name '*.ipynb' -exec jupyter nbconvert --to notebook --execute --inplace --allow-errors {} \;

