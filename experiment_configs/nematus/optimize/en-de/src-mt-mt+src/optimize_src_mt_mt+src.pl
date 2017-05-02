#!/usr/bin/env perl

use strict;
use POSIX;
use File::Temp qw/ tempfile tempdir /;

my $PID = $$;
$SIG{TERM} = $SIG{INT} = $SIG{QUIT} = sub { die; };

use Getopt::Long;
use File::Spec;

# TODO: call constrained decoding with all necessary args at each MERT iteration
my $GBS_DIR = "/home/chris/projects/constrained_decoding";
my $MOSES_DIR = "/home/chris/projects/mosesdecoder/bin";
my $MOSES_SCRIPTS = "/home/chris/projects/mosesdecoder/scripts";

my $SRC_MODEL_DIR = "/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe";
my $MT_MODEL_DIR = "/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/mt-pe";
my $MT_SRC_MODEL_DIR = "/media/1tb_drive/nematus_ape_experiments/ape_qe/en-de_models/en-de_mt_aligned/fine_tune/model";
my $DEV_DATA_DIR = "/media/1tb_drive/Dropbox/data/qe/ape/concat_wmt_2016_2017";

my $DEV_REF = "$DEV_DATA_DIR/dev.pe";

# Note filename must be run1.dense
my $START_WEIGHTS = "/home/chris/projects/qe_sequence_labeling/experiment_configs/nematus/optimize/en-de/src-mt-mt+src/run1.dense";

my $NBEST = 20;
my $BEAM_SIZE = 20;

my $time = time();
my $WORK = "tuning.$time";
my $SCORER = "BLEU";

my $MAX_IT = 10;

GetOptions(
    "w|working-dir=s" => \$WORK,
    "m|moses-bin-dir=s" => \$MOSES_DIR,
    "s|scorer=s" => \$SCORER,
    "i|maximum-iterations=i" => \$MAX_IT,
);

my $MIRA = "$MOSES_DIR/kbmira";
my $MERT = "$MOSES_DIR/mert";
my $EVAL = "$MOSES_DIR/evaluator";
my $EXTR = "$MOSES_DIR/extractor";

my $CONFIG = "--sctype $SCORER";

$WORK = File::Spec->rel2abs($WORK);

execute("mkdir -p $WORK");
# start with a default run1.dense (don't call the `amunmt show weights`)
execute("cp $START_WEIGHTS $WORK");

#execute("$AMUNN $DECODER_OPTS --show-weights > $WORK/run1.dense");

# uses default run1.dense to create run1.initopt
dense2init("$WORK/run1.dense", "$WORK/run1.initopt");

execute("rm -rf $WORK/progress.txt");
for my $i (1 .. $MAX_IT) {
    unless(-s "$WORK/run$i.out") {
#        execute("cat $DEV_SRC | $AMUNN $DECODER_OPTS --load-weights $WORK/run$i.dense --n-best | perl -pe 's/\@\@ //g' > $WORK/run$i.out");
        # TODO: add weight file parsing and MT output postprocessing to this command
        # TODO: implement --load_weights file parsing
#        sed 's/\@\@ //g' | \
#$mosesdecoder/scripts/recaser/detruecase.perl | \
#$mosesdecoder/scripts/tokenizer/deescape-special-chars.perl
        execute("python $GBS_DIR/scripts/translate_nematus.py -m $MT_SRC_MODEL_DIR/model.npz $SRC_MODEL_DIR/model.iter370000.npz $MT_MODEL_DIR/model.iter290000.npz -c $MT_SRC_MODEL_DIR/model.npz.json $SRC_MODEL_DIR/model.iter370000.npz.json $MT_MODEL_DIR/model.iter290000.npz.json -i $DEV_DATA_DIR/dev.mt.factor_corpus $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped --nbest $NBEST --beam_size $BEAM_SIZE --load_weights $WORK/run$i.dense --mert_nbest | sed 's/\@\@ //g' | $MOSES_SCRIPTS/recaser/detruecase.perl | $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl > $WORK/run$i.out");
#        python $GBS_DIR/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.iter370000.npz $MT_MODEL_DIR/model.iter290000.npz  -c $SRC_MODEL_DIR/model.iter370000.npz.json $MT_MODEL_DIR/model.iter290000.npz.json  -i $DEV_DATA_DIR/dev.src.prepped.small $DEV_DATA_DIR/dev.mt.prepped.small --nbest 5 --beam_size 10 --mert_nbest --load_weights /media/1tb_drive/nematus_ape_experiments/ape_qe/test_mert_optimization/tuning.1493403939/run1.dense
#        python $GBS_DIR/scripts/translate_nematus.py -m $SRC_MODEL_DIR/model.iter370000.npz $MT_MODEL_DIR/model.iter290000.npz  -c $SRC_MODEL_DIR/model.iter370000.npz.json $MT_MODEL_DIR/model.iter290000.npz.json  -i $DEV_DATA_DIR/dev.src.prepped $DEV_DATA_DIR/dev.mt.prepped --nbest 5 --beam_size 10 --mert_nbest
    }
    execute("$EVAL $CONFIG --reference $DEV_REF -n $WORK/run$i.out | tee -a $WORK/progress.txt");
    
    my $j = $i + 1;
    unless(-s "$WORK/run$j.dense") {
        execute("$EXTR $CONFIG --reference $DEV_REF -n $WORK/run$i.out -S $WORK/run$i.scores.dat -F $WORK/run$i.features.dat");
        
        my $SCORES = join(",", map { "$WORK/run$_.scores.dat" } (1 .. $i));
        my $FEATURES = join(",", map { "$WORK/run$_.features.dat" } (1 .. $i));

        # Note: the -d argument is the number of dimensions == the number of features
        execute("$MERT --sctype $SCORER --scfile $SCORES --ffile $FEATURES --ifile $WORK/run$i.initopt -d 3 -n 20 -m 20 --threads 20 2> $WORK/mert.run$i.log");

        log2dense("$WORK/mert.run$i.log", "$WORK/run$j.dense");
        dense2init("$WORK/run$j.dense", "$WORK/run$j.initopt");
    }
    execute("cp $WORK/run$j.dense $WORK/weights.txt")
}

sub execute {
    my $command = shift;
    logMessage("Executing:\t$command");
    my $ret = system($command);
    if($ret != 0) {
        logMessage("Command '$command' finished with return status $ret");
        logMessage("Aborting and killing parent process");
        kill(2, $PID);
        die;
    }
}

sub log2dense {
    my $log = shift;
    my $dense = shift;

    open(OLD, "<", $log) or die "can't open $log: $!";
    open(NEW, ">", $dense) or die "can't open $dense: $!";
    
    my @weights;
    while(<OLD>) {
        chomp;
        if (/^Best point: (.*?)  =>/) {
            @weights = split(/\s/, $1);
        }
    }
    close(OLD) or die "can't close $log: $!";
    my $i = 0;
    foreach(@weights) {
        print NEW "F$i= ", $_, "\n";
        $i++;
    }
    close(NEW);
}

sub dense2init {
    my $dense = shift;
    my $init = shift;

    open(OLD, "<", $dense) or die "can't open $dense: $!";
    open(NEW, ">", $init) or die "can't open $init: $!";
    
    my @weights;
    while(<OLD>) {
        chomp;
        if (/^M\d+= (\S*)$/) {
            push(@weights, $1);
        }
    }
    close(OLD) or die "can't close $dense: $!";
    print NEW join(" ", @weights), "\n";
    print NEW "0 " x scalar @weights, "\n";
    print NEW "1 " x scalar @weights, "\n";
    close(NEW);
}


sub logMessage {
    my $message = shift;
    my $time = POSIX::strftime("%m/%d/%Y %H:%M:%S", localtime());
    my $log_message = $time."\t$message\n"; 
    print STDERR $log_message;
}

sub wc {
    my $path = shift;
    my $lineCount = `wc -l < '$path'` + 0;
    return $lineCount;
}
