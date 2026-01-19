// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/vector.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <vector>

using namespace dealii;

int
main()
{
  const unsigned int n_blocks = 4;

  using DBV = LinearAlgebra::distributed::DynamicBlockVector<double>;

  DBV vec(n_blocks);
  for (unsigned int b = 0; b < vec.n_blocks(); ++b)
    vec.block(b).reinit(b + 1);
  vec.collect_sizes();

  std::cout << "Number of blocks initially: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';

  // Number blocks according to their pointers to check their positions later
  std::vector<std::shared_ptr<DBV::BlockType>> blocks;

  // Blocks order: 0, 1, 2, 3
  for (unsigned int i = 0; i < vec.n_blocks(); ++i)
    blocks.push_back(vec.block_ptr(i));

  // Insert new block
  std::cout << "\nCheck block insertion\n";
  const unsigned int insert_position = 1;

  auto new_block = std::make_shared<DBV::BlockType>(5);
  vec.insert_block(new_block, insert_position);

  std::cout << std::boolalpha;

  std::cout << "Number of blocks after insert: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';

  // Check positions of the blocks, blocks order: 0, new, 1, 2, 3
  std::cout << "Is block 0 correct: " << (vec.block_ptr(0) == blocks[0])
            << '\n';
  std::cout << "Is block 1 correct: "
            << (vec.block_ptr(insert_position) == new_block) << '\n';
  std::cout << "Is block 2 correct: " << (vec.block_ptr(2) == blocks[1])
            << '\n';
  std::cout << "Is block 3 correct: " << (vec.block_ptr(3) == blocks[2])
            << '\n';
  std::cout << "Is block 4 correct: " << (vec.block_ptr(4) == blocks[3])
            << '\n';

  // Delete a block - #2
  std::cout << "\nCheck block removal\n";
  const unsigned int remove_position = 3;

  auto old_block = vec.remove_block(remove_position);

  std::cout << "Is removed block correct: " << (old_block == blocks[2]) << '\n';

  std::cout << "Number of blocks after remove: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';

  // Check positions of the blocks, blocks order: 0, new, 1, 3
  std::cout << "Is block 0 correct: " << (vec.block_ptr(0) == blocks[0])
            << '\n';
  std::cout << "Is block 1 correct: " << (vec.block_ptr(1) == new_block)
            << '\n';
  std::cout << "Is block 2 correct: " << (vec.block_ptr(2) == blocks[1])
            << '\n';
  std::cout << "Is block 3 correct: " << (vec.block_ptr(3) == blocks[3])
            << '\n';

  // Move the block downwards
  std::cout << "\nCheck block move downwards\n";
  const unsigned int move_down_from = 0;
  const unsigned int move_down_to   = 2;

  vec.move_block(move_down_from, move_down_to);

  std::cout << "Number of blocks after move down: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';

  // Check positions of the blocks, blocks order: new, 1, 0, 3
  std::cout << "Is block 0 correct: " << (vec.block_ptr(0) == new_block)
            << '\n';
  std::cout << "Is block 1 correct: " << (vec.block_ptr(1) == blocks[1])
            << '\n';
  std::cout << "Is block 2 correct: " << (vec.block_ptr(2) == blocks[0])
            << '\n';
  std::cout << "Is block 3 correct: " << (vec.block_ptr(3) == blocks[3])
            << '\n';

  // Move the block upwards
  std::cout << "\nCheck block move upwards\n";
  const unsigned int move_up_from = 3;
  const unsigned int move_up_to   = 1;

  vec.move_block(move_up_from, move_up_to);

  std::cout << "Number of blocks after move up: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';

  // Check positions of the blocks, blocks order: new, 3, 1, 0
  std::cout << "Is block 0 correct: " << (vec.block_ptr(0) == new_block)
            << '\n';
  std::cout << "Is block 1 correct: " << (vec.block_ptr(1) == blocks[3])
            << '\n';
  std::cout << "Is block 2 correct: " << (vec.block_ptr(2) == blocks[1])
            << '\n';
  std::cout << "Is block 3 correct: " << (vec.block_ptr(3) == blocks[0])
            << '\n';

  // Add 2 new blocks by simply resizing the vector
  std::cout << "\nCheck blocks addition when resizes to grow\n";

  const unsigned int n_add = 2;
  vec.reinit(vec.n_blocks() + n_add);

  std::cout << "Number of blocks after reinit: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';
  std::cout << "Is block 0 correct: " << (vec.block_ptr(0) == new_block)
            << '\n';
  std::cout << "Is block 1 correct: " << (vec.block_ptr(1) == blocks[3])
            << '\n';
  std::cout << "Is block 2 correct: " << (vec.block_ptr(2) == blocks[1])
            << '\n';
  std::cout << "Is block 3 correct: " << (vec.block_ptr(3) == blocks[0])
            << '\n';
  std::cout << "Is block 4 correct: " << (vec.block_ptr(4) != nullptr) << '\n';
  std::cout << "Is block 5 correct: " << (vec.block_ptr(5) != nullptr) << '\n';

  // Remove 3 blocks by simpyl resizing the vector
  std::cout << "\nCheck blocks removal when resizes to shrink\n";

  const unsigned int n_remove = 3;
  vec.reinit(vec.n_blocks() - n_remove);

  std::cout << "Number of blocks after reinit: " << vec.n_blocks() << '\n';
  std::cout << "Total size: " << vec.size() << '\n';
  std::cout << "Is block 0 correct: " << (vec.block_ptr(0) == new_block)
            << '\n';
  std::cout << "Is block 1 correct: " << (vec.block_ptr(1) == blocks[3])
            << '\n';
  std::cout << "Is block 2 correct: " << (vec.block_ptr(2) == blocks[1])
            << '\n';
}