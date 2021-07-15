#ifndef CSVRow_H
#define CSVRow_H

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVRow
{
    public:
        CSVRow(char sep = ',');

        std::string operator[](std::size_t index) const;
        std::size_t size() const;
        std::istream& readNextRow(std::istream& str);
        friend std::istream& operator>>(std::istream& str, CSVRow &row){
            row.readNextRow(str);
            return str;
        }

    private:
        std::string         m_line;
        std::vector<int>    m_data;
        char sep;
};
#endif