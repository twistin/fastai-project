---
title: "Is it a velutina or a crabro wasps? My second train model."
description: I will explain my learning process and experience with my second contact with my own training model of lesson 2, Production. 
toc: true
comments: true
layout: post
categories: [fastai,vision]
image: images/fastpages_posts/actions/velut.jpeg
author: sdCarr
---


I have just started the second lesson of the Fastai course. In this case, the question was to make a model that predicts whether an image is a velutine wasp or a goat wasp. And then put the model into production using Hugging Spaces. 

The first thing I did was, I downloaded a 
test image of a velutine wasp using the ddg() function:

```
ims = search_images_ddg('wasp vespa velutina')
len(ims)

dest = 'images/velutina.jpg'.
download_url(ims[0], dest, show_progress= True)

im = Image.open(dest)
im.to_thumb(150,150)
```
 