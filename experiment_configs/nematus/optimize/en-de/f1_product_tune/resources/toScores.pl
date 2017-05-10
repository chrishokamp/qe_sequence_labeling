use strict;

my $c = 0;
print "SCORES_TXT_BEGIN_0 $c 12 4 QE\n";
while(<STDIN>) {
  if(($. - 1)  % 12 == 0 and $. > 1) {
     $c++;
     print "SCORES_TXT_END_0\n";
     print "SCORES_TXT_BEGIN_0 $c 12 4 QE\n";
  }
  print $_;
}
print "SCORES_TXT_END_0\n";
