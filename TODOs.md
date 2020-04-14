# ARCHITECTURE

## TODO 
change class structure: objectives into FeatureObjectives(get_layer) & ImageObjectives()
--> for transparent images (i.e. incl alpha channel)

## TODO
several images at the same time if init from image

## TODO
potentially: rethink whole architecture with render - obj classes. Maybe something more elegant?

## TODO
basic testing


## TODO
objectives - there still might be bugs when optimizing more than one img at the same time:
check that there's no gradient leakage between those images


# EXPLORATION

## TODO
try other learning policies; e.g. OneCycle, multiple cycles, or other LR Schedulers?
maybe even different lrs for different pixels? e.g. in center different than outside?

## TODO
try out different color models; especially HSL, or maybe even fix SL (e.g. from a real image) and then only change hue (different learning rates for H, S, and L)? -> walk cyclically through hue? check out different hue parametrizations, such as Munsell, NCS

## TODO
how will training the networks already in HSL (or other color space) change feature visualizations?

## TODO:
 what happens if we add zeroth/first layer + later layer objective? shouldn't this make sharp but meaningful images?
 ---> doesn't seem to work well! doesn't make anything more sharp

## TODO:
ImageObjective that penalize init from image images to look more like the original image (eg in ssim)(needs changed class structure)

## TODO
combine optimization with CAM: focus feature visualization on those parts of an image that the cam shows to be most important for the clf

# EXPLORATION - TRAINING

## TODO
find image patches that most highly activate channels instead of optimize! --> everywhere instead of fv? (first tests somewhat discouraging - needs another stricter try)

# General
## TODO
use transformation library such as kornia or nvidia-dali instead of own ones (first benchmark; a lot of stuff is prob better on GPU than on CPU)

## TODO
include proper transformations for visualizations --> they should match what we trained with, not weird resizes
-> check everywhere, e.g. in initializer as well

## TODO
dataset specific decorrelation! (see beginning of `img_param.py`)

# INTERFACE & EXPORT

## TODO
make feature importances more intuitive, e.g. via bars

## TODO
whole export stuff is a mess --> clean up or better write new from scratch

## TODO
whole web interface!

# PORT LUCID

## TODO 
alpha channel! (start - `alpha_stuff.py`
 -> clean everything up, currently via list and everything's ugly
 -> background image should be way smoother
 -> gamma correction
 -> regularize alpha \*(1-mean(alpha))
 -> check lucid implementation again, they might have other changes



## TODO
diversity!
--> check whether current impl is correct -> looks different than in tf and doesn't work well at all?
--> this for `ChannelDiversity`, `ConvNeuronDiversity` and `FCDiversity`, and potentially then also for others

## TODO
lots of other stuff. e.g.
    - check whether DirectionChannel works well
    - more objectives
    - multiply two objectives
    - differential feature parametrizations, ...


# ATTRIBUTION

## TODO
better/other alternatives to CAM (or optimizations to CAM)
those right now are shitty and also sometimes constant...

## TODO
include guided backprop & occlusion maps from other scripts here

# GWAS

## TODO
also show association between each neuron and genotype --> gives even more interpretability

## TODO:
create many new images via feature visualization and then --> can we use them for statistics?
