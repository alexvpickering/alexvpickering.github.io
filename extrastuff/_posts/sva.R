library(ggplot2)
library(reshape)
library(RColorBrewer)
library(sva)
library(gridExtra)

# Legend Function ---------------------------------

get_legend<-function(myggplot){
    tmp <- ggplot_gtable(ggplot_build(myggplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)
}

# Plot Function ---------------------------------

# plot the expression values
plot_heatmap <- function(y, legend) {
    
    ym <- melt(y, varnames=c("gene", "sample"))
    
    # get colours from a red-yellow-blue palette
    pal <- colorRampPalette(rev(brewer.pal(11, "RdYlBu")))
    
    ggplot(ym, aes(y=gene, x=sample, fill=value)) +
        geom_tile() + 
        scale_fill_gradientn(name="", colours=pal(100), 
                             limits=c(-6, 6), guide=legend) + 
        theme_minimal(base_size=9) + 
        scale_y_discrete(breaks=NULL) +
        scale_x_discrete(breaks=NULL) +
        labs(x=NULL, y=NULL)
        
}


# Setup ---------------------------------


# random matrix of expression values
y <- matrix(rnorm(10000), nrow=1000, ncol=10)
colnames(y) <- paste(rep(c("ctrl", "test"), each=5), 1:5)

# add some signal
y[1:300, 6:10] <- 
    y[1:300, 6:10] + 2

sig <- plot_heatmap(y, "none")

# add heterogeneity
y[200:400, c(2, 3, 5, 10)] <-
    y[200:400, c(2, 3, 5, 10)] - 2

het <- plot_heatmap(y, "none")

# SVA ----------------------------------

group <- factor(rep(c("ctl", "test"), each = 5))
mod  <- model.matrix(~0 + group)
colnames(mod) <- c("ctl", "test")
mod0 <- model.matrix(~1, data = group)

svobj <- sva(y, mod, mod0)

# remove effect of discovered surrogate variables
yc <- crossmeta:::clean_y(y, mod, svobj$sv)
sv <- plot_heatmap(yc, "legend")


# Plot Together -------------------------

# get legend
legend <- get_legend(sv)

# remove legend
sv  <- sv  + theme(legend.position="none")


mult <- arrangeGrob(sig, het, sv, legend, ncol=4, widths=c(3, 3, 3, 0.8))


ggsave("heatmapsv.svg", mult, width=10, height=5, bg="#f8f8f8")
