/*
 *   Copyright (c) 2007 John Weaver
 *   Copyright (c) 2016 Gluttton <gluttton@ukr.net>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#if !defined(_MATRIX_H_)
#define _MATRIX_H_

#include "munkres-cpp/matrix_base.h"
#include <initializer_list>
#include <cassert>

namespace munkres_cpp {

template<class T> class Matrix : public matrix_base<T> {
public:
	Matrix() = default;
	Matrix(size_t, size_t);
	Matrix(const std::initializer_list<std::initializer_list<T>> &);
	Matrix(const Matrix<T> &);
	Matrix<T> &operator=(const Matrix<T> &);
	~Matrix() override;
	// All operations modify the matrix in-place.
	void resize(size_t, size_t, T default_value = T(0)) override;
	const T &operator()(size_t x, size_t y) const override
	{
		assert(x < m_rows);
		assert(y < m_columns);
		assert(m_matrix != nullptr);
		return m_matrix[x][y];
	};
	T &operator()(size_t x, size_t y) override
	{
		assert(x < m_rows);
		assert(y < m_columns);
		assert(m_matrix != nullptr);
		return m_matrix[x][y];
	};
	size_t columns() const override { return m_columns; }
	size_t rows() const override { return m_rows; }

private:
	T **m_matrix{nullptr};
	size_t m_rows{0};
	size_t m_columns{0};
};

template<class T>
Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T>> &init) : Matrix<T>()
{
	m_rows = init.size();
	if (m_rows) {
		m_columns = init.begin()->size();
		if (m_columns > 0) {
			resize(m_rows, m_columns);
		}
	}

	size_t i = 0;
	for (auto row = init.begin(); row != init.end(); ++row, ++i) {
		assert(row->size() == m_columns);
		size_t j = 0;
		for (auto value = row->begin(); value != row->end(); ++value, ++j) {
			m_matrix[i][j] = *value;
		}
	}
}

template<class T> Matrix<T>::Matrix(const Matrix<T> &other) : Matrix<T>()
{
	if (other.m_matrix) {
		resize(other.m_rows, other.m_columns);
		std::copy(other.m_matrix[0], other.m_matrix[0] + m_rows * m_columns, m_matrix[0]);
	}
}

template<class T> Matrix<T>::Matrix(size_t rows, size_t columns) : Matrix<T>()
{
	resize(rows, columns);
}

template<class T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
{
	if (this == &other) {
		return *this;
	}

	if (other.m_matrix != nullptr) {
		resize(other.m_rows, other.m_columns);
		std::copy(other.m_matrix[0], other.m_matrix[0] + m_rows * m_columns, m_matrix[0]);
	} else {
		// Free arrays.
		delete[] m_matrix[0];
		delete[] m_matrix;

		m_matrix = nullptr;
		m_rows = 0;
		m_columns = 0;
	}

	return *this;
}

template<class T> Matrix<T>::~Matrix()
{
	if (m_matrix != nullptr) {
		delete[] m_matrix[0];
		delete[] m_matrix;
	}
	m_matrix = nullptr;
}

template<class T> void Matrix<T>::resize(size_t rows, size_t columns, T default_value)
{
	T **new_matrix = new T *[rows]; // Row pointers.
	try {
		new_matrix[0] = new T[rows * columns]; // All data in one stripe.
	} catch (std::bad_alloc &) {
		delete[] new_matrix;
		throw;
	}
	for (size_t i = 1; i < rows; i++) {
		new_matrix[i] = new_matrix[0] + i * columns;
	}
	std::fill(new_matrix[0], new_matrix[0] + rows * columns, default_value);

	if (m_matrix) {
		// Copy data from saved pointer to new arrays.
		const size_t minrows = std::min(rows, m_rows);
		const size_t mincols = std::min(columns, m_columns);
		for (size_t y = 0; y < mincols; y++) {
			for (size_t x = 0; x < minrows; x++) {
				new_matrix[x][y] = m_matrix[x][y];
			}
		}

		// Delete old arrays.
		if (m_matrix != nullptr) {
			delete[] m_matrix[0];
			delete[] m_matrix;
		}
	}

	m_rows = rows;
	m_columns = columns;
	m_matrix = new_matrix;
}

} // namespace munkres_cpp

#endif /* !defined(_MATRIX_H_) */
