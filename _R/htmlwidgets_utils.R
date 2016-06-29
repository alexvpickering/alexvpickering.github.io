knit_htmlwidgets <- function(input,
                             output_dir = "./_includes/htmlwidgets",
                             ...) {
    
    file_name <- rev(unlist(strsplit(input, split = "/")))[1]
    path <- rmarkdown::render(input, "html_document", output_dir = output_dir)
    remove_doctype(path)
    
}

remove_doctype <- function(input) {
    
    html_lines <- readLines(input)
    keep <- grep("^<!DOCTYPE html>$", html_lines, invert = TRUE)
    writeLines(html_lines[keep], input)
    
}



