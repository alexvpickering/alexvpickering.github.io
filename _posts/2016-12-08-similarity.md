---
layout: post
title: Transcription Profile Similarity
date: 2016-12-8
description: Exploring how overlap is computed in ccmap.
ogimage: "/img/mds_thumb.png"
---


Introduction
-------
<br>
A core issue in gene expression analysis is determining the similarity between two different gene expression profiles. The approach that I took for this task in <a href="http://bioconductor.org/packages/ccmap/" target="blank">ccmap</a> provides a measure of similarity that effectively weights genes based on their extent of differential expression. With this approach, there is no need to choose an arbitrary number of top differentially expressed genes when computing similarity. 

<br>

Similarity Metric
-------
<br>

To calculate the ccmap similarity metric, gene expression signatures are first sorted based on their magnitude of differential expression and labeled as to their direction of differential expression (up or down regulated).


{% highlight r %}
library(data.table)

# drug and query differential gene expression signatures with 4 genes
dr_sig  <- c(0.5, 1.5, -3, -0.5)
qy_sig  <- c(1.5, -0.5, -1, -2)
names(dr_sig) <- names(qy_sig) <- toupper(letters[1:4])

# order by absolute differential expression values
dr_sig <- dr_sig[order(abs(dr_sig), decreasing=TRUE)]
#    C    B    A    D 
# -3.0  1.5  0.5 -0.5 

qy_sig <- qy_sig[order(abs(qy_sig), decreasing=TRUE)]
#    D    A    C    B 
# -2.0  1.5 -1.0 -0.5 

# replace effect size values with sign
dr_sig <- sign(dr_sig)
#  C  B  A  D 
# -1  1  1 -1

qy_sig <- sign(qy_sig)
#  D  A  C  B 
# -1  1 -1 -1 
{% endhighlight %}

Ordering the signatures captures the extent of differential expression: the more differentially expressed a gene, the sooner it appears in the vector. Genes that differ minimally between control and treatment samples appear later in the vector. The direction of differential expression (up or down) has been captured by the sign (+1 or -1). 

Next, we create a matrix that captures the similarity between the two profiles. For an increasing number of considered query genes, we determine the net overlap for an increasing number of considered drug genes. For example, the first row of the net overlap matrix consists of three 0's followed by a 1 because overlap between the first most differentially expressed query gene (D) doesn't occur until the fourth most differentially expressed drug gene (D). The second row of the overlap matrix is determined by considering the first two most differentially expressed query genes (D & A). As such, the net overlap increases from zero to one after considering the third drug gene (A) and from one to two after all four drug genes have been considered. The same process continues for the last two rows, with net overlap increasing or decreasing by one depending on whether a gene is regulated in the same or the opposite direction in the drug and query signatures.


{% highlight r %}
net_overlap <- t(matrix(c(0, 0, 0, 1,
                          0, 0, 1, 2,
                          1, 1, 2, 3,
                          1, 0, 1, 2), nrow=4))
{% endhighlight %}

This matrix captures the net overlap between two signatures at all possible values for the number of considered genes between those two signatures. For example, cell (4,4) = 2 encodes the fact that, when all four query and drug genes are considered, three genes (C, A, and D) are regulated in the same directions and one gene (B) is regulated in opposite directions (net overlap = 3-1 = 2).

The net overlap matrix can be considered as providing the height values for the overlap surface between two transcriptional signatures. The volume under this overlap surface (the sum of the net overlap matrix), provides a measure of similarity (used by ccmap) that incorporates both whether or not genes are regulated in the same direction as well as the similarity in their magnitudes of regulation. To see this second point, consider the net overlap matrix of a signature with itself.


{% highlight r %}
self_overlap <- t(matrix(c(1, 1, 1, 1,
                           1, 2, 2, 2,
                           1, 2, 3, 3,
                           1, 2, 3, 4), nrow=4))
{% endhighlight %}

Viewed as the heights of a surface, this matrix rises up as fast as possible, thus maximizing the volume underneath it:

<img src="/img/self_2000.png" srcset="/img/self_2000.png 2000w, /img/heatmap_4000.png 4000w" class="ImageBorder ImageResponsive">


Compare this to the net overlap matrix of two signatures whose genes are regulated in the same direction (all genes up/down regulated similarly) but with the opposite absolute order of differential expression. 


{% highlight r %}
rev_overlap <- t(matrix(c(0, 0, 0, 1,
                          0, 0, 1, 2,
                          0, 1, 2, 3,
                          1, 2, 3, 4), nrow=4))
{% endhighlight %}

This second matrix doesn't begin its elevation until the last possible moment. As such, for a matrix that reaches the same final height by cell (4, 4), the volume under its surface is as small as possible:

<img src="/img/rev_2000.png" srcset="/img/rev_2000.png 2000w, /img/heatmap_4000.png 4000w" class="ImageBorder ImageResponsive">

This volume under the surface measure of similarity effectively weights genes based on their extent of differential expression. Genes with minimal differential expression are only considered at higher drug/query gene numbers and thus contribute less to the total volume under the surface.

<br>

Algorithmic Efficiency
-------
<br>

While the ccmap measure of similarity has the desirable properties above, it presents an apparent practical issue: memory and computational burden. For example, the memory footprint of an overlap matrix that is 20,000 by 20,000 (approximately the number of human genes) is ~3.2GB. Thankfully, both the memory and computational issues are tractable.

In order demonstrate this, first consider the matrix that simply records the positions and directions of overlap between the query and drug signatures (incidence matrix):



{% highlight r %}
# incidence matrix for original query and drug signatures
incidence_mat <- t(matrix(c(0, 0, 0, 1,
                            0, 0, 1, 0,
                            1, 0, 0, 0,
                            0,-1, 0, 0), nrow=4))
{% endhighlight %}

The net overlap matrix can be derived from the incidence matrix by first recording the cumulative sum for the rows of the incidence matrix, followed by the cumulative sum for the columns of the resulting matrix:


{% highlight r %}
# row cumulative sum of incidence matrix
rowCumsum <- t(matrix(c(0, 0, 0, 1,
                        0, 0, 1, 1,
                        1, 1, 1, 1,
                        0,-1,-1,-1), nrow=4))

# column cumulative sum of rowCumsum matrix
rowcolCumsum <- t(matrix(c(0, 0, 0, 1,
                           0, 0, 1, 2,
                           1, 1, 2, 3,
                           1, 0, 1, 2), nrow=4))

# all.equal(rowcolCumsum, net_overlap)
#[1] TRUE
{% endhighlight %}

The incidence matrix is sparse (mostly 0 valued: only need to record non-zero values/indices). Additionally, I was able to derive a closed form solution for calculating the sum of the values in the rowcolCumsum (aka net overlap) matrix directly from the sparse incidence matrix. As a result, the volume under the surface similarity metric between two 20,000 gene signatures can be directly calculated from the sparse incidence matrix that requires ~320 kB of RAM (instead of ~3.2GB of RAM if using the net overlap matrix). For example:


{% highlight r %}
# non-zero values of incidence matrix
x <- c(1, 1, 1, -1)

# non-zero row indices 
i <- c(1, 2, 3, 4)

# non-zero column indices
j <- c(4, 3, 1, 2)

# sum of rowcolCumsum matrix
vos <- sum(x * (max(i) - i + 1) * (max(j) - j + 1))

# all.equal(vos, sum(net_overlap))
# [1] TRUE
{% endhighlight %}

<br>

Summary
-------
<br>
In this post I used a simple example to explain the ccmap similarity metric and demonstrate its desirable features. The ccmap similarity metric considers both the overlap in the direction and extent of differential expression. Importantly, this metric weights both query and drug genes according to their extent of differential expression and thereby eliminates the need to choose a fixed number of query or drug genes. Instead, ccmap determines the overlap for all combinations of query and drug gene numbers simultaneously and computes a composite volume under the surface measure of similarity. Although a straightforward implementation of the ccmap similarity metric is computationally burdensome, a closed-form sparse matrix approach was derived which provides massive speed-ups and reductions in memory usage.
