#include "CSVRange.h"

CSVRange::CSVRange(std::istream& str, char sep_arg)
    : stream(str), sep(sep_arg)
{}
CSVIterator CSVRange::begin() const {return CSVIterator{stream, sep};}
CSVIterator CSVRange::end()   const {return CSVIterator{};}