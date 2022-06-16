# FastAI curso - Jeremy Howard

![](/Users/silvinodiazcarreras/Desktop/Captura%20de%20pantalla%202022-05-12%20a%20las%2012.04.45.png)

### Módulos Fastai

from fastai .vision.all import *

from fastai . tabular.all import *

from  *fastai*.*text*.*all import* *

from fastai.collab import *

from fastai.callback.hook import *

from fastbook import *

from fastai.vision.widgets import *

doc() es una función especial para obtener documentación de alguna función 

p ej  

```python
doc(untar_data)
```

y me devolverá 

```python
untar_data(url, archive=None, data=None, c_key='data', force_download=False)
Download `url` to `fname` if `dest` doesn't exist, and extract to folder `dest`

To get a prettier result with hyperlinks to source code and documentation, install nbdev: pip install nbdev
```

### Para descargar imégenes de Bing Search

```python
!pip install bing_image_downloader
from bing_image_downloader import downloader
downloader.download('string de búsqueda', limit=100, output_dir='dataset')
```

### Para descargar imágenes de Google

```python
pip install Google-Images-Search
from google_images_search import GoogleImagesSearch
# you can provide API key and CX using arguments,
# or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX
gis = GoogleImagesSearch('your_dev_api_key', 'your_project_cx')

# define search params
# option for commonly used search param are shown below for easy reference.
# For param marked with '##':
#   - Multiselect is currently not feasible. Choose ONE option only
#   - This param can also be omitted from _search_params if you do not wish to define any value
_search_params = {
    'q': '...',
    'num': 10,
    'fileType': 'jpg|gif|png',
    'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
    'safe': 'active|high|medium|off|safeUndefined', ##
    'imgType': 'clipart|face|lineart|stock|photo|animated|imgTypeUndefined', ##
    'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge|imgSizeUndefined', ##
    'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow|imgDominantColorUndefined', ##
    'imgColorType': 'color|gray|mono|trans|imgColorTypeUndefined' ##
}

# this will only search for images:
gis.search(search_params=_search_params)

# this will search and download:
gis.search(search_params=_search_params, path_to_dir='/path/')

# this will search, download and resize:
gis.search(search_params=_search_params, path_to_dir='/path/', width=500, height=500)

# search first, then download and resize afterwards:
gis.search(search_params=_search_params)
for image in gis.results():
    image.url  # image direct url
    image.referrer_url  # image referrer url (source) 

    image.download('/path/')  # download image
    image.resize(500, 500)  # resize downloaded image

    image.path  # downloaded local file path
```

### Para descargar desde DDG

```python
!pip install -q jmd_imagescraper

from pathlib import Path
root = Path().cwd()/"images"

from jmd_imagescraper.core import *
duckduckgo_search(root, "Cats", "cute kittens", max_results=100)
duckduckgo_search(root, "Dogs", "cute puppies", max_results=100)


from jmd_imagescraper.imagecleaner import *
display_image_cleaner(root)
```

# Lección 1

### Shortcuts jupyter notebook

- m:: Convert cell to Markdown
- y:: Convert cell to Code
- d+d:: Delete cell
- o:: Toggle between hide or show output
- Shift+Arrow up/Arrow down:: Select multiple cells. Once you have selected them you can operate on them like a batch (run, copy, paste etc).
- Shift+M:: Merge selected cells
- `Ctrl + Shift + -`dividirá la celda actual en dos desde donde está el cursor.

### Magic lines (jupyter)

Las línea mágicas   son funciones que se pueden ejecutar en las celdas. Deben estar al principio de una línea y tomar como argumento el resto de la línea desde donde se llaman. Se llaman colocando un signo '%' antes del comando. Las más útiles son:  

%matplotlib inline:: Asegura que todos los trazados de matplotlib serán trazados en la celda de salida dentro del cuaderno y se mantendrán en el cuaderno cuando se guarden.  
Este comando se llama siempre junto al principio de cada cuaderno del curso fast.ai.

%timeit:: Ejecuta una línea diez mil veces y muestra el tiempo medio de ejecución.

%debug: Inspecciona una función que está mostrando un error utilizando el depurador de Python. Si escribes esto en una celda justo después de un error, serás dirigido a una consola donde podrás inspeccionar los valores de todas las variables.

### Mirando los datos

Vamos a utilizar la función untar_data a la que debemos pasar una URL como argumento y que descargará y extraerá los datos.

Para descargar cualquiera de los conjuntos de datos o pesos preentrenados, simplemente ejecute [`untar_data`](https://docs.fast.ai/data.external.html#untar_data)pasando cualquier nombre de conjunto de datos mencionado anteriormente de la siguiente manera:

```python
#Cada cuaderno comienza con las siguientes tres líneas; 
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#éstas garantizan que cualquier edición de las bibliotecas 
#que realices se recargue aquí automáticamente, y 
#también que cualquier gráfico o imagen que se muestre, 
#lo haga en este cuaderno.





'''Importamos todos los paquetes necesarios. Vamos a trabajar con la librería fastai V1 que se asienta sobre Pytorch 1.0. La biblioteca fastai proporciona muchas funciones útiles que nos permiten construir rápida y fácilmente redes neuronales y entrenar nuestros modelos.'''
#help(untar_data)
#doc(untar_data)
from google_images_download import google_images_download
path = untar_data(URLs.PETS); path
help(google_images_download)
response = google_images_download.googleimagesdownload()
arguments = {"keywords":"Angry human expression","limit":5, "print_urls":True}
paths = response.download(arguments) 
path.ls()
path_anno = path/'annotations'
path_img = path/'images'
```

```python
#id first_training
#caption Results from the first training
# CLICK ME
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images' #untar_data() desgargará conjunto de 
# datos

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

#### Cómo usar *untar_data* si el conjunto de datos proviene de una una fuente externa:

```python
url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
path = download_data(url)
path.as_posix()
# '/home/gg/.fastai/archive/flower_photos.tgz'

data = untar_data(path.as_posix()) # or pass str(path)
data
# Path('/home/gg/.fastai/data/flower_photos')
```

### Usar path en carpeta local

```python
# create path
path2 = Path("/content/sample_data")

# check if path is directory
print(path2.is_dir())
```

# Lección 2

Primera línea de código del cuaderno:

```python
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
'''
“!” → le dice al cuaderno Jupyter que pase la línea restante a bash shell

[ -e /content ] → es el comando en bash para verificar si existe 
un archivo o directorio y devuelve verdadero o falso. 
Entonces, aquí estamos verificando si existe un directorio 
llamado contenido en la ruta /contenido (/ es ​​el directorio raíz
 y el contenido estaría justo debajo de él si existe)

&& → significa "Si el último comando devolvió True, 
ejecute el siguiente comando". Así que solo verificamos si existe 
o no un directorio /contenido. Si existe, pasaremos a ejecutar 
el comando que viene después de "&&" ya que 
la salida de [ -e /content] sería "verdadero"

pip install -Uqq fastbook → instale el paquete fastbook y actualícelo (-U) 
si ya existe. -qq significa estar muy muy callado?
'''
```
