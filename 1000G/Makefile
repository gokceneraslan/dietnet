ORIG=ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz
ORIG_TBI=$(ORIG).tbi
ORIG_PANEL=affy_samples.20141118.panel
TEMP=genotmp
TEMP_GROUP=$(TEMP).bed $(TEMP).bim $(TEMP).fam
OUT=genotypes
OUT_GROUP=$(OUT).bed $(OUT).bim $(OUT).fam
OUT_NPY=$(OUT)_x.npy $(OUT)_y.npy

all: finalize removetemp

$(ORIG):
	#download genotypes in vcf format
	wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/$(ORIG)
	wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/$(ORIG_TBI)

$(ORIG_PANEL):
	#download sample info
	wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/$(ORIG_PANEL)

$(TEMP_GROUP): $(ORIG) $(ORIG_PANEL)
	# easy way to install via conda: conda install -c bioconda plink2
	plink2 --vcf $(ORIG) --maf 0.05 --geno 0 --not-chr X Y MT \
		   --indep-pairwise 500 5 0.5 --threads 4 --make-bed --out $(TEMP)

$(OUT_GROUP): $(TEMP_GROUP)
	#exclude snps in high LD
	plink2 --bfile $(TEMP) --exclude $(TEMP).prune.out --threads 4 --make-bed --out $(OUT)

$(OUT_NPY): $(OUT_GROUP)
	#install plinkio, numpy and pandas prior to this: pip install plinkio numpy pandas
	python ../plink2numpy.py -f $(OUT) -p $(ORIG_PANEL)

.PHONY: removetemp
removetemp:
	-rm -rf $(TEMP)*

.PHONY: clean
clean: removetemp
	-rm -rf $(ORIG) $(ORIG_TBI) $(ORIG_PANEL) $(OUT)*

download: $(ORIG) $(ORIG_PANEL)
preprocess: $(OUT_GROUP)
finalize: $(OUT_NPY)
