$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
$max_repeat = 5;
@default_files = ('thesis.tex');
$aux_dir = 'build';
$out_dir = '.';
