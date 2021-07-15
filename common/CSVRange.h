#ifndef CSVRange_H
#define CSVRange_H

#include "CSVIterator.h"

class CSVRange
{
    std::istream&   stream;
    public:
        CSVRange(std::istream& str);
        CSVIterator begin() const;
        CSVIterator end()   const;
};

#endif