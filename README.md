# HAHA humor tweet classifier

HAHA humor tweet classifier is a project to identify the humor mechanism of a tweet (in spanish).

As a pre activity of the HAHA competition of 2021, a little workshop from Udelar - Fing has been developed in order to explore these data. There are 2 main tasks:
* identify mechanism (12 categories - excluyents)
* identify targets (12 main categories, each with differents sub categories - tree structure -)

The scope of this project is to identify the first feature.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install gensim = 3.8.1;
pip install pandas = 1.2.3;
pip install numpy = 1.20.2;
pip install keras = 2.4.3;
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
#### Execution
```python
>>> import data_preparation_module
>>> preprocess_data()
```

### Training module
Module where models(s) are builded and trained.
1. Build model
2. Load embedding
3. Train
4. Perist model's metrics

#### Execution
```bash
$ python3 training_module.py $model_id $model_name $activation_function $recurrent_function $layer_units
```

### Base line module
Module where a dummy model is trained in order to obtain a baseline.

#### Execution
```bash
$ python3 line_base_classifier.py
```

### Test data module
Module where test (unknow category) is classify by the BEST MODEL. Those classification are sended to the responsables? of the workshop, to be checked with the real data.

#### Execution
```bash
$ python3 test_data_classifier.py
```

## Further work
From the software end, it would be great to refactor the model construction using a proper framework to release the solution as an API. Then, add some techniques to increase the performance, such as threads and process managment.

From the NPL end, there is a great way to go still. On one hand, there is to increase the quantity of the combinations tested, in order to obtain the better configuration.
On the other hand, it must be considered the alternative to deprecated this model and use a pre trained one, for example [BERTO](https://blog.google/products/search/search-language-understanding-bert/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgement

- Computer science of Facultad de Ingeniería ([INCO](https://www.fing.edu.uy/inco/grupos/pln/))

Specially, Santiago Gongora and Luis Chiruzzo as coordinators of the workshop.

## License
[MIT](https://choosealicense.com/licenses/mit/)


