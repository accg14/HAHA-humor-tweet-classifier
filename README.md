# HAHA humor tweet classifier

HAHA humor tweet classifier is a project to identify the humor mechanism of a tweet (in spanish).

As a pre activity of the HAHA competition of 2021, a little workshop from Udelar - fing has been developed in order to explore these data. There are 2 main tasks:
* identify mechanism (12 categories - excluyents)
* identify targets ()

the scope of this project is to identify the first feature.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install gensim=
pip install pandas
pip install numpy
pip install keras
```

## Usage

### Data preparation module
Module where row data is extracted and sanitized, the process includes:
1. Extract data from source (CSV)
2. Sanitize the data
    1. Remove unwanted chars
    2. Remove user mentions (a.k.a '@username')
    3. Remove symbols
    4. Splitt over spaces
3. Compute tweet's words median
4. Generate embedding
5. Persist 2 & 3 to files

### Training module
Module where models(s) are builded and trained.
1. Build model
2. Load embedding
3. Train
4. Perist model's metrics

### Base line module
Module where a dummy model is trained in order to obtain a baseline.

### Test data module
Module where test (unknow category) is classify by the BEST MODEL. Those classification are sended to the responsables? of the workshop, to be checked with the real data.


```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
