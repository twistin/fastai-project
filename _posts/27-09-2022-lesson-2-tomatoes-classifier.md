---
title: "Tomatoes Classifier"
description: After completing lesson two for second time, I set out to create a more complex model than the previous one I had made to classify wasps. 
The idea was to create a tomato sorter. 
toc: true
comments: true
layout: post
categories: [fastai,vision]
image: images/fastpages_posts/actions/actions_logo.png
author: sdCarr
---

After completing lesson two, I set out to create a more complex model than the previous one I had made to classify wasps. 
The idea was to create a tomato sorter. I set to work with the data. The truth is that I found many varieties of tomatoes, some of them very similar because they are variants of the same species. This made it very difficult to approximate the prediction of the model. I decided to choose 18 varieties that were sufficiently different to try to get the model as close as possible, even though I knew how difficult it was, as some very similar tomato varieties were distinguished by colour, for example. The varieties chosen were the following:
zebra tomato', 'yellow pear tomato', 'hillbilly tomato', 'green zebra tomato', 'great white tomato', 'Giulietta tomato', 'Garden Peach tomato', 'cherokee tomato', 'campari tomato','Krim tomato','Azoychka tomato','cherry tomato','Grape Tomatoes','roma tomato','red beefsteak tomato','Green tomatoes','Kumato Tomatoes','heirloom tomato'.