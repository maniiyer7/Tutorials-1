#! /usr/bin/perl -w
# kjv.pl - Implement FULLTEXT search of King James Version of the Bible

use strict;
use warnings;
use lib qw(/usr/local/apache/lib/perl);
use CGI qw(:standard escapeHTML);
use DBI;
use Cookbook;

# Change history:
# 2002-11-01
# - Add phrase searching and record-count-only options.

# Note: Need MySQL 4.x server for support of shorter indexed words

my $dbh = Cookbook::connect ();

my $title = "KJV Search";
my $page = header ()
           . start_html (-title => $title, -bgcolor => "white")
           . p (strong ("King James Bible Search Form"))
           . p ("This is a very rudimentary search engine for the King James\n"
           . "version of the Bible.\n"
           . "(It serves as a simple demonstration of MySQL's basic\n"
           . "FULLTEXT capabilities.)\n");

$page .= make_search_form ($dbh);

$page .= end_html ();

my $phrase = param ("phrase");

if (defined ($phrase) && $phrase =~ /\S/) # perform search
{
  $page .= perform_search ($dbh);
}

$page .= hr ()
         . "Comments or questions? Contact "
         . a ({-href => "mailto:paul\@kitebird.com"}, "Paul DuBois")
         . "."
         . br ();

print $page;

$dbh->disconnect ();

# ----------------------------------------------------------------------

sub make_search_form
{
my $dbh = shift;
my $page;
my ($val_ref, $label_ref) = get_book_info ($dbh);

  $page .= start_form (-action => url ())
           . table (
               Tr (
                 td ("What to search:"),
                 td (popup_menu (-name => "book",
                                 -values => $val_ref,
                                 -labels => $label_ref)
                 )
               ),
               Tr (
                 td ("Word or words to search for:"),
                 td (textfield (-name => "phrase", -size => 60))
               ),
               Tr (
                 td ("Treat words as phrase:"),
                 td (radio_group (-name => "as_phrase",
                                  -values => [ "1", "0" ],
                                  -labels => { "1" => "yes", 0 => "no" },
                                  -default => "0")
                 )
               ),
               Tr (
                 td ("Return count only:"),
                 td (radio_group (-name => "count_only",
                                  -values => [ "1", "0" ],
                                  -labels => { "1" => "yes", 0 => "no" },
                                  -default => "0")
                 )
               )
           )
           . br ()
           . submit (-name => "choice", -value => "Search")
           . end_form ();
  return ($page);
}

# Return references to array of book values and hash that maps values
# to labels.  For regular books, the value is the book number and the
# label is the name.  The initial entries are ALL/Entire Bible, NT/New
# Testament, and OT/Old Testament.

sub get_book_info
{
my $dbh = shift;
my @value;
my %label;

  # Install the initial entries.  Use label keys that are different
  # than the book numbers (which are always integers).

  push (@value, "ALL", "OT", "NT");
  $label{ALL} = "Entire Bible";
  $label{OT} = "Old Testament";
  $label{NT} = "New Testament";

  my $sth = $dbh->prepare (qq{
              SELECT bnum, bname
              FROM kjv_bookmap ORDER BY bnum
            });
  $sth->execute ();
  while (my ($num, $name) = $sth->fetchrow_array ())
  {
    push (@value, $num);
    $label{$num} = $name;
  }
return (\@value, \%label);
}

sub perform_search
{
my $dbh = shift;
my $book = param ("book") || "";
my $phrase = param ("phrase") || "";
my $as_phrase = param ("as_phrase") || 0;
my $count_only = param ("count_only") || 0;
my $page = "";
my @placeholder;
my $where;

  # return unless some non-empty phrase was given
  return "" unless defined ($phrase) && $phrase =~ /\S/;

  if ($as_phrase)
  {
    # this might break if phrase contains " characters...?
    $phrase = "\"$phrase\"";
    $where = "WHERE MATCH(vtext) AGAINST(? IN BOOLEAN MODE)";
  }
  else
  {
    $where = "WHERE MATCH(vtext) AGAINST(?)";
  }
  push (@placeholder, $phrase);

  # Check scope of seach.  Look for the special cases first.
  # If it's none of them, assume it's a book number.

  if ($book eq "ALL")
  {
    # no action needed
  }
  elsif ($book eq "OT")
  {
    $where .= " AND bsect = 'O'";
  }
  elsif ($book eq "NT")
  {
    $where .= " AND bsect = 'N'";
  }
  else  # book number
  {
    $where .= " AND bnum = ?";
    push (@placeholder, $book);
  }

  # Run COUNT(*) query to determine number of matches.
  # Then run the query that returns the rows (maximum of 100).
  # If there are more than 100, modify the "number of hits" message

  my $count = $dbh->selectrow_array (qq{
                SELECT COUNT(*)
                FROM kjv
                $where
              }, undef, @placeholder);
  if ($count_only)
  {
    $page = p ("$count records found") . $page;
    return ($page);
  }
  my $rows = 0;
  my $sth = $dbh->prepare (qq{
              SELECT bname, cnum, vnum, vtext
              FROM kjv
              $where
              ORDER BY bnum, cnum, vnum
              LIMIT 100
            });
  $sth->execute (@placeholder);
  while (my ($bookname, $cnum, $vnum, $vtext) = $sth->fetchrow_array ())
  {
    ++$rows;
    $page .= p (escapeHTML ("$bookname $cnum:$vnum $vtext"));
  }
  if ($rows >= $count)
  {
    $page = p ("$rows records found") . $page;
  }
  else
  {
    $page = p ("Displaying first $rows of $count matches") . $page;
  }
  return ($page);
}

sub error
{
my $msg = shift;

  print p (escapeHTML ($msg));
  exit (0);
}
