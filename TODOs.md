## ARCHITECTURE

### proper package structure with setup etc

### change class structure

split objectives into FeatureObjectives(get_layer) & ImageObjectives()
--> what's a good framework for ImageObjectives? - should enable transparent images (i.e. incl alpha channel) - should enable smoothing/blurring of images in each step (not as TFM, but actually smoothing)

### enable several images at the same time for init from image

### rethink whole architecture

with render - obj classes. Maybe something more elegant?

### basic testing

### objectives

there still might be bugs when optimizing more than one img at the same time:
check that there's no gradient leakage between those images

## EXPLORATION

### learning policies

try other learning policies; e.g. OneCycle, multiple cycles, or other LR Schedulers?
maybe even different lrs for different pixels? e.g. in center different than outside?

### color models

try out different color models; especially HSL, or maybe even fix SL (e.g. from a real image) and then only change hue (different learning rates for H, S, and L)? -> walk cyclically through hue? check out different hue parametrizations, such as Munsell, NCS

### sharper imgs?

what happens if we add zeroth/first layer + later layer objective? shouldn't this make sharp but meaningful images?
---> doesn't seem to work well! doesn't make anything more sharp in first try

### make images more real

ImageObjective that penalize init from image images to look more like the original image (eg in ssim)(needs changed class structure)

### quantitative evaluation of visualizations

What kind of metrics to compare different visualization strategies?

### Combine attribution & visualization

combine optimization with CAM: focus feature visualization on those parts of an image that the cam shows to be most important for the clf

## EXPLORATION - TRAINING

### color space in training

how will training the networks already in HSL (or other color space) change feature visualizations?

### different training schemes

how do different training approaches (#epochs, learning rate & policy, data augmentation, overfit vs underfit, ...) change the visualizations?
--> is there any prior work on this?? couldn't find anything interesting...

### image patches in training/testing data set

find image patches that most highly activate channels instead of optimize! --> everywhere instead of fv? (first tests somewhat discouraging - needs another stricter try)

## General

### augmentation library

use transformation library such as kornia or nvidia-dali instead of own ones (first benchmark; a lot of stuff is prob better on GPU than on CPU)

### check tfms

include proper transformations for visualizations --> they should match what we trained with, not weird resizes
-> check everywhere, e.g. in initializer as well

### check decorrelation

dataset specific decorrelation! (see beginning of `img_param.py`)

## INTERFACE & EXPORT

### whole web interface!

### whole export stuff is a mess

--> clean up or better write new from scratch

## PORT LUCID

### transparency channel

alpha channel! (start - `alpha_stuff.py`
-> clean everything up, currently via list and everything's ugly
-> background image should be way smoother
-> gamma correction
-> regularize alpha \*(1-mean(alpha))
-> check lucid implementation again, they might have other changes

### diversity

--> check whether current impl is correct -> looks different than in tf and doesn't work well at all?
--> this for `ChannelDiversity`, `ConvNeuronDiversity` and `FCDiversity`, and potentially then also for others

### lots of other stuff.

e.g. - check whether DirectionChannel works well - more objectives - multiply two objectives - differential feature parametrizations, ...

## ATTRIBUTION

### improve/debug CAM

vis right now are shitty and also sometimes constant...

### include guided backprop & occlusion maps from other scripts here

## GWAS

### associations?

also show association between each neuron and genotype --> gives even more interpretability

### stats?

create many new images via feature visualization and then --> can we use them for statistics?
