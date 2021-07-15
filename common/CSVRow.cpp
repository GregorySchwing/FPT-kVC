#include "CSVRow.h"

CSVRow::CSVRow(char sep_arg) : sep(sep_arg){}

std::string CSVRow::operator[](std::size_t index) const
{
    return std::string(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
}
std::size_t CSVRow::size() const
{
    return m_data.size() - 1;
}
std::istream& CSVRow::readNextRow(std::istream& str)
{
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    //while((pos = m_line.find(',', pos)) != std::string::npos)
    while((pos = m_line.find(sep, pos)) != std::string::npos)
    {
        m_data.emplace_back(pos);
        ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos   = m_line.size();
    m_data.emplace_back(pos);
    return str;
}