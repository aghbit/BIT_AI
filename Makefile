KEY=
TIMEOUT=3600
REGEX='$(shell echo $(KEY) | sed 's/.ipynb//').*.ipynb'
TREEGREP=tree -fi | grep $(REGEX)
LATEX=jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to latex 
PDF=pdflatex
REQUIREMENTS=requirements.txt


# makes pdfs from all notebooks whose name match the regex (so all by default)
all: $(shell $(TREEGREP) | sed 's/ipynb/pdf/g')

install:
	pip install -r $(REQUIREMENTS)

# lists all notebooks whose name matches the regex
list:
	$(TREEGREP)

# removes unstaged files
clean: stripout
	git clean -xdf	

stripout:
	nbstripout `$(TREEGREP)`

# updates requirements.txt
requpdate:
	pip freeze > $(REQUIREMENTS)	

%.pdf: %.tex
	cd `dirname $^` && $(PDF) `basename $^`

%.tex: %.ipynb
	$(LATEX) $^ --output=`basename $@` 	

