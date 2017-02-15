---
layout: post
title: Parkin Mimics
date: 2016-11-27
description: Using the R packages crossmeta and ccmap to find mitophagy mimics.
---


Introduction
------------
<br>
Like oxygen? Me too. It might seem strange, but oxygen can actually be quite toxic. A prime example - its appearance on earth is responsible for a massive extinction event ~2.3 billion years ago (see <a href="https://en.wikipedia.org/wiki/Great_Oxygenation_Event" target="blank">The Great Oxygenation Event</a>, aka the Oxygen Catastrophe, Oxygen Crisis, Oxygen Holocaust, etc). Remarkably, our single celled ancestors devised a means to capture the deadly power of this newfound oxygen and use it to drive the biomolecular machines of life.

Slight correction - our unicellular ancestors didn't figure out breathing themselves, but rather formed a partnership of sorts with another cell that did. These other cells, now called mitochondria, provided the surplus of energy that allowed our ancestors to graduate from their unicellular existence. This idea, that mitochondria were once independant cells, explains why mitochondria have a distinct genome and can replicate inside of their host cells. 

Crucially, the proximity of DNA and oxygen use that happens inside mitochondria puts the mitochondrial genome in danger of damage. The resulting dysfunction of mitochondria may contribute to aging. One way that our cells appear to guard against mitochondrial dysfunction is by selectively eliminating damaged mitochondria. In a <a href="http://www.nature.com/articles/ncomms13100" target="blank">recent study</a> led by Nikolay P. Kandul, it was demonstrated that overexpressing the Parkin or PINK1 gene promotes this garbaging (word?) of mitochondria.

<br>

Parkin Mimics
--------------
<br>
Until genetic overexpression becomes feasible in humans, Parkin activation will have to occur through pharmacological means. I couldn't find any known Parkin activators, so I thought I would check the ~27000 drug/perturbagen microarray signatures available in the ccdata package to see what produces the most comparable transcriptional signature to that caused by Parkin overexpression. If you would like to follow along, you will need R, Rstudio, and the Bioconductor packages <a href="http://bioconductor.org/packages/crossmeta/" target="blank">crossmeta</a>, <a href="http://bioconductor.org/packages/ccmap/" target="blank">ccmap</a>, and <a href="http://bioconductor.org/packages/ccdata/" target="blank">ccdata</a>.

To start, I searched <a href="https://www.ncbi.nlm.nih.gov/geo/" target="blank">GEO</a> for _"Parkin"_ and found two microarray datasets where Parkin was overexpressed. I then used crossmeta to perform a meta-analysis of these two datasets as follows:



{% highlight r %}
library(crossmeta)

# create directory for data
dir.create('PARKIN'); setwd('PARKIN')

# get/load raw data
gse_names <- c('GSE61973', 'GSE29494')
get_raw(gse_names)
esets <- load_raw(gse_names)

# differential expression analysis
anals <- diff_expr(esets)

# meta analysis
es <- es_meta(anals, cutoff=1)
dprimes <- get_dprimes(es)
{% endhighlight %}

The next steps use ccmap to query the ~230000 drug/perturbagen transcriptional signatures from the Connectivity Map and LINCS L1000 datasets. 

Note: The development version of ccmap and ccdata are required to query the L1000 dataset.



{% highlight r %}
library(ccmap)

# queries
top_cmap <- query_drugs(dprimes$meta, 'cmap_es')
top_l1000 <- query_drugs(dprimes$meta, 'l1000_es')


head(top_cmap)

# copper sulfate:  0.296  
# tacrine:         0.232       
# menadione:       0.217
# Prestwick-1080:  0.217
# metrizamide:     0.214
# novobiocin:      0.210
{% endhighlight %}

TADA! Potential Parkin overexpression mimics. I say _mimics_, rather than activators, because the above drugs don't necessarily activate Parkin directly. Rather, they cause transcriptional changes that are most similar to those caused by Parkin overexpression. Details, details.

Now if only I had a research lab and money ... then I could do some biology. Sad face.
