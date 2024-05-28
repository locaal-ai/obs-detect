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

#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include <algorithm>
#include <cmath>
#include <forward_list>
#include <limits>

namespace munkres_cpp {

template<typename T>
static constexpr typename std::enable_if<std::is_integral<T>::value, bool>::type is_zero(const T &v)
{
	return v == 0;
}

template<typename T>
static constexpr typename std::enable_if<!std::is_integral<T>::value, bool>::type
is_zero(const T &v)
{
	return FP_ZERO == std::fpclassify(v);
}

template<typename T, template<typename> class M> class Munkres {
public:
	Munkres(M<T> &);
	Munkres(const Munkres &) = delete;
	Munkres &operator=(const Munkres &) = delete;

private:
	bool find_uncovered_in_matrix(size_t &, size_t &) const;
	int step1();
	int step2();
	int step3();
	int step4();
	int step5();
	int step6();

	const size_t size;
	M<T> &matrix;
	M<char> mask_matrix;
	bool *const row_mask;
	bool *const col_mask;
	size_t saverow;
	size_t savecol;
	enum MASK : char {
		NORMAL,
		STAR // starred,
		,
		PRIME // primed.
	};
};

template<typename T, template<typename> class M>
void minimize_along_direction(M<T> &matrix, bool over_columns)
{
	// Look for a minimum value to subtract from all values along the "outer" direction.
	size_t i = 0, j = 0, size = matrix.rows();
	size_t &r = over_columns ? j : i;
	size_t &c = over_columns ? i : j;
	for (; i < size; i++, j = 0) {
		T min = matrix(r, c);

		// As long as the current minimum is greater than zero, keep looking for the minimum.
		// Start at one because we already have the 0th value in min.
		for (j = 1; j < size && min > 0; j++)
			min = std::min(min, matrix(r, c));

		if (min > 0)
			for (j = 0; j < size; j++)
				matrix(r, c) -= min;
	}
}

template<typename T, template<typename> class M>
bool Munkres<T, M>::find_uncovered_in_matrix(size_t &row, size_t &col) const
{
	for (col = 0; col < size; col++)
		if (!col_mask[col])
			for (row = 0; row < size; row++)
				if (!row_mask[row])
					if (is_zero(matrix(row, col)))
						return true;

	return false;
}

template<typename T, template<typename> class M> int Munkres<T, M>::step1()
{
	for (size_t row = 0; row < size; row++) {
		for (size_t col = 0; col < size; col++) {
			if (is_zero(matrix(row, col))) {
				for (size_t nrow = 0; nrow < row; nrow++)
					if (STAR == mask_matrix(nrow, col))
						goto next_column;

				mask_matrix(row, col) = STAR;
				goto next_row;
			}
		next_column:;
		}
	next_row:;
	}

	return 2;
}

template<typename T, template<typename> class M> int Munkres<T, M>::step2()
{
	size_t covercount = 0;

	for (size_t col = 0; col < size; col++)
		for (size_t row = 0; row < size; row++)
			if (STAR == mask_matrix(row, col)) {
				col_mask[col] = true;
				covercount++;
			}

	return covercount >= size ? 0 : 3;
}

template<typename T, template<typename> class M> int Munkres<T, M>::step3()
{
	// Main Zero Search
	// 1. Find an uncovered Z in the distance matrix and prime it. If no such zero exists, go to Step 5
	// 2. If No Z* exists in the row of the Z', go to Step 4.
	// 3. If a Z* exists, cover this row and uncover the column of the Z*. Return to Step 3.1 to find a new Z
	if (find_uncovered_in_matrix(saverow, savecol)) {
		mask_matrix(saverow, savecol) = PRIME; // Prime it.
		for (size_t ncol = 0; ncol < size; ncol++) {
			if (mask_matrix(saverow, ncol) == STAR) {
				row_mask[saverow] = true; // Cover this row and
				col_mask[ncol] =
					false; // uncover the column containing the starred zero
				return 3;      // repeat.
			}
		}
		return 4; // No starred zero in the row containing this primed zero.
	}

	return 5;
}

template<typename T, template<typename> class M> int Munkres<T, M>::step4()
{
	// Seq contains pairs of row/column values where we have found
	// either a star or a prime that is part of the ``alternating sequence``.
	// Use saverow, savecol from step 3.
	std::forward_list<std::pair<size_t, size_t>> seq{{saverow, savecol}};

	// Increment Set of Starred Zeros
	// 1. Construct the ``alternating sequence'' of primed and starred zeros:
	//   Z0     : Unpaired Z' from Step 4.2
	//   Z1     : The Z* in the column of Z0
	//   Z[2N]  : The Z' in the row of Z[2N-1], if such a zero exists
	//   Z[2N+1]: The Z* in the column of Z[2N]
	// The sequence eventually terminates with an unpaired Z' = Z[2N] for some N.
	size_t dim[] = {0, savecol};
	const char mask[] = {STAR, PRIME};
	for (size_t i = 0; dim[i] < size; ++dim[i]) {
		if (mask_matrix(dim[0], dim[1]) == mask[i]) {
			// We have to find these two pairs: z1 and z2n.
			seq.push_front({dim[0], dim[1]});
			i = (i + 1) & 1; // Switch dimension.
			dim[i] = -1;     // After increment this value becames zero.
		}
	}

	for (const auto &i : seq) {
		// 2. Unstar each starred zero of the sequence.
		if (mask_matrix(i.first, i.second) == STAR)
			mask_matrix(i.first, i.second) = NORMAL;

		// 3. Star each primed zero of the sequence,
		// thus increasing the number of starred zeros by one.
		if (mask_matrix(i.first, i.second) == PRIME)
			mask_matrix(i.first, i.second) = STAR;
	}

	// 4. Erase all primes, uncover all columns and rows,
	std::replace(mask_matrix.begin(), mask_matrix.end(), PRIME, NORMAL);
	std::fill_n(row_mask, size, false);
	std::fill_n(col_mask, size, false);

	// and return to Step 2.
	return 2;
}

template<typename T, template<typename> class M> int Munkres<T, M>::step5()
{
	// New Zero Manufactures
	// 1. Let h be the smallest uncovered entry in the (modified) distance matrix.
	// 2. Add h to all covered rows.
	// 3. Subtract h from all uncovered columns
	// 4. Return to Step 3, without altering stars, primes, or covers.
	T h = std::numeric_limits<T>::max();
	for (size_t col = 0; col < size; col++)
		if (!col_mask[col])
			for (size_t row = 0; row < size; row++)
				if (!row_mask[row])
					if (h > matrix(row, col) && !is_zero(matrix(row, col)))
						h = matrix(row, col);

	for (size_t row = 0; row < size; row++)
		if (row_mask[row])
			for (size_t col = 0; col < size; col++)
				matrix(row, col) += h;

	for (size_t col = 0; col < size; col++)
		if (!col_mask[col])
			for (size_t row = 0; row < size; row++)
				matrix(row, col) -= h;

	return 3;
}

// Linear assignment problem solution
// [modifies matrix in-place.]
// matrix(row,col): row major format assumed.
//
// Assignments are remaining 0 values
// (extra 0 values are replaced with 1)
template<typename T, template<typename> class M>
Munkres<T, M>::Munkres(M<T> &matrix)
	: size{std::max(matrix.rows(), matrix.columns())},
	  matrix{matrix},
	  mask_matrix{size, size},
	  row_mask{new bool[size]},
	  col_mask{new bool[size]},
	  saverow{0},
	  savecol{0}
{
	const size_t rows = matrix.rows();
	const size_t columns = matrix.columns();

	std::fill_n(row_mask, size, false);
	std::fill_n(col_mask, size, false);

	if (rows != columns)
		// If the input matrix isn't square, make it square and fill
		// the empty values with the maximum possible value.
		matrix.resize(size, size, std::numeric_limits<T>::max());

	// Prepare the matrix values...
	minimize_along_direction(matrix, rows >= columns);
	minimize_along_direction(matrix, rows < columns);

	// Follow the steps
	int step = 1;
	while (step) {
		switch (step) {
		case 1:
			step = step1(); // step is always 2
					// Fallthrough.
		case 2:
			step = step2(); // step is always either 0 or 3
			break;
		case 3:
			step = step3(); // step in [3, 4, 5]
			break;
		case 4:
			step = step4(); // step is always 2
			break;
		case 5:
			step = step5(); // step is always 3
			break;
		}
	}

	// Store results.
	for (size_t col = 0; col < size; col++)
		for (size_t row = 0; row < size; row++)
			matrix(row, col) = (T)(mask_matrix(row, col) == STAR ? 0 : 1);

	// Remove the excess rows or columns that we added to fit the input to a square matrix.
	matrix.resize(rows, columns, 1);

	delete[] row_mask;
	delete[] col_mask;
}

} // namespace munkres_cpp

#endif /* !defined(_MUNKRES_H_) */
