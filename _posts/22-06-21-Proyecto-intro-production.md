---
title: "My first training model: Predicting if an image is a Jazz Bass, Precision or Mustang bass."
description: Publication of my first prediction model from the first two lessons of the Fastai course.
toc: true
comments: true
layout: post
categories: [actions, markdown, 01_intro.ipynb]
image: images/fastpages_posts/actions/actions_logo.png
author: sdCarr
---

## Estado del aprendizaje profundo 

First of all, I must clarify that I have started the Fastai course in the spring of 2022, which obviously means that I am a complete newcomer in deep learning and my knowledge is very scarce and limited.
When addressing the state of deep learning, I had to make it clear that my knowledge on the subject is from the moment in which the course I am taking was published, that is, the year 2020 and not right now, since in the course notebooks it is on that date when they are made public. By this I mean that most likely, almost two years later, the panorama has changed substantially given the speed with which events occur in this field of Artificial Intelligence (From now on, *AI*).

Deep learning is good at recognizing where objects are in an image and can highlight their locations and name each object found. This is known as **object detection**.

Deep learning algorithms are generally not good at recognizing images that are significantly different in structure or style than the ones used to train the model. For example, if there were no black and white images in the training data, the model might perform poorly on black and white images. Similarly, if the training data does not contain hand-drawn images, the model is likely to perform poorly with hand-drawn images. There is no general way to check what image types are missing from your training set, but we will show in this chapter some ways to try to recognize when unexpected image types arise in the data when the model is used in production (this is known as data check *outside domain*).

A major challenge for object detection systems is that image tagging can be time consuming and expensive. One approach that is particularly useful is to synthetically generate variations of the input images, such as rotating them or changing their brightness and contrast; this is called **data augmentation** and it also works well for text and other types of models. We will discuss it in detail in this chapter.

Another point to consider is that even though your problem may not seem like a computer vision problem, it might be possible with a bit of imagination to turn it into one. For example, if you are trying to classify sounds, you might try to convert the sounds into images of their acoustic waveforms, and then train a model on those images.

Four fields where deep learning is useful:

- Computer vision
  
- Texro (natural language processing)
  
- Combine text and images
  
- Tabulated data
  
- Recommendation systems

## Project
My project was based on testing my knowledge acquired in reading the first two lessons of the course as well as the corresponding videos.
My objective was to create a small application that would be able to discriminate if a photo of an electric bass instrument was a Fender brand, and Precision, Mustang or Jazz Bass models.

### Actions
The actions for such a simple project were to search for and download a set of bass images large enough to make a dataset with which to train the model.
To do this, I downloaded, with the help of the search_images_bing function, a batch of 150 images of each searched word. This is the number of images that Bing allows to download by default with a free account.
To download images with BIS we must open an account in Azure and then create a Bing image Search app. Once this is done we must obtain an access key.

Once copied, we must relocate it inside the cell as the second argument of the `os.environ.get()` function.

```python
key = os.environ.get('AZURE_SEARCH_KEY', '956c67d521be415899bb0664b8cad97a')
```

Once you have set the key, you can use `search_images_bing`. This functionality is provided by the small class of utilities included in the online notebooks. If you're not sure where a function is defined, you can write it in your notebook to find out:

```python
search_images_bing
```

`<function fastbook.search_images_bing>`

```python
results = search_images_bing(key, 'bass fender precision')
ims = results.attrgot('contentUrl')
len(ims)
```

`150` # returns 150 results by default Bing

We've successfully downloaded the URLs of 150 grizzly bears (or at least the images that Bing Image Search finds for that search term).

```python
dest = 'images/bass fender precision.jpg'
download_url(ims[0], dest)
```

`101.92% [327680/321508 00:00<00:00]`

`Path('images/bass fender precision.jpg')`

Let's now look at one of these downloaded images in the search

```python
im = Image.open(dest)

im.to_thumb(128,128)
```

We are now going to use fastai's `download_images` to download all the URLs for each of our search terms. We'll put each one in a separate folder:

```python
bass_types = 'fender precision bass','fender jazz bass','fender mustang bass'
path = Path('bass_carpet')
```

```python
if not path.exists():
    path.mkdir()
    for or in bass_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bass')
        download_images(dest, urls=results.attrgot('contentUrl'))
```

Our folder has image files, as expected:

```python
fns = get_image_files(path)
fns
```

And that will return us an array with the search terms:

`#433) [Path('bass_carpet/fender bass precision/00000000.jpg'),Path('guitars/fender bass precision/00000107.jpg'),Path('guitars/fender bass precision/00000073.jpg'), Path('guitars/fender bass precision/00000027.jpg'),Path('guitars/fender bass precision/00000071.jpg'),Path('guitars/fender bass precision/00000092.jpg'),Path('guitars/ fender bass precision/00000034.jpg'),Path('guitars/fender bass precision/00000078.jpg'),Path('guitars/fender bass precision/00000050.jpg'),Path('guitars/fender bass precision/00000059 .jpg')...]`

If we want to verify the downloaded images we use the following function:

```python
failed = verify_images(fns)
failed
```

To remove all failed images, you can use unlink on each of them.

```python
failed.map(Path.unlink);
```