# latexmk configuration: teaches it to run makeglossaries so that
#   latexmk -pdf main.tex
# builds the document, bibliography and acronym list in one command.

$pdf_mode = 1;
$bibtex_use = 2;          # run bibtex and clean .bbl on -C

# --- glossaries / acronyms -------------------------------------------
add_cus_dep('acn', 'acr', 0, 'makeglossaries');
add_cus_dep('glo', 'gls', 0, 'makeglossaries');

sub makeglossaries {
    my ($base_name, $path) = fileparse($_[0]);
    pushd($path);
    my $return = system "makeglossaries", $base_name;
    popd();
    return $return;
}

# Remove generated glossary files with "latexmk -C".
push @generated_exts, 'acn', 'acr', 'alg', 'glg', 'glo', 'gls', 'ist';
