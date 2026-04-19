"""Unit tests for utils.py"""

import os
import tempfile
import shutil
from utils import create_train_test_split, extract_number


def test_number_extraction():
    """Test that numbers are extracted correctly from filenames."""
    assert extract_number('img-000001.jpg') == 1
    assert extract_number('img13labels-000001.png') == 1  # Last number, not first
    assert extract_number('image123.png') == 123
    assert extract_number('nonum.jpg') == 0
    print("✓ Number extraction tests passed")


def test_file_matching():
    """Test that image and label files match by number."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    try:
        source_rgb = os.path.join(temp_dir, 'rgb')
        source_label = os.path.join(temp_dir, 'label')
        train_rgb = os.path.join(temp_dir, 'train_rgb')
        train_label = os.path.join(temp_dir, 'train_label')
        test_rgb = os.path.join(temp_dir, 'test_rgb')
        test_label = os.path.join(temp_dir, 'test_label')
        
        os.makedirs(source_rgb)
        os.makedirs(source_label)
        
        # Create test files matching user's example
        # Image: img-000001.jpg, Label: img13labels-000001.png
        with open(os.path.join(source_rgb, 'img-000001.jpg'), 'w') as f:
            f.write('dummy image')
        with open(os.path.join(source_label, 'img13labels-000001.png'), 'w') as f:
            f.write('dummy label')
        
        # Add more test files
        with open(os.path.join(source_rgb, 'img-000002.jpg'), 'w') as f:
            f.write('dummy image 2')
        with open(os.path.join(source_label, 'img13labels-000002.png'), 'w') as f:
            f.write('dummy label 2')
        
        # This should work without raising an error
        train_files, test_files = create_train_test_split(
            source_rgb=source_rgb,
            source_label=source_label,
            train_rgb=train_rgb,
            train_label=train_label,
            test_rgb=test_rgb,
            test_label=test_label,
            train_ratio=0.5,
            seed=42
        )
        
        print(f"✓ File matching test passed")
        print(f"  Train files: {train_files}")
        print(f"  Test files: {test_files}")
        
        # Verify files were copied
        assert len(os.listdir(train_rgb)) > 0, "Training RGB directory is empty"
        assert len(os.listdir(train_label)) > 0, "Training label directory is empty"
        print("✓ Files copied successfully")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_missing_label():
    """Test that missing label raises appropriate error."""
    temp_dir = tempfile.mkdtemp()
    try:
        source_rgb = os.path.join(temp_dir, 'rgb')
        source_label = os.path.join(temp_dir, 'label')
        train_rgb = os.path.join(temp_dir, 'train_rgb')
        train_label = os.path.join(temp_dir, 'train_label')
        test_rgb = os.path.join(temp_dir, 'test_rgb')
        test_label = os.path.join(temp_dir, 'test_label')
        
        os.makedirs(source_rgb)
        os.makedirs(source_label)
        
        # Create image without matching label
        with open(os.path.join(source_rgb, 'img-000001.jpg'), 'w') as f:
            f.write('dummy image')
        
        # This should raise FileNotFoundError
        try:
            create_train_test_split(
                source_rgb=source_rgb,
                source_label=source_label,
                train_rgb=train_rgb,
                train_label=train_label,
                test_rgb=test_rgb,
                test_label=test_label,
                train_ratio=0.5,
                seed=42
            )
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            print(f"✓ Missing label error test passed: {e}")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    print("Running utils.py tests...\n")
    test_number_extraction()
    test_file_matching()
    test_missing_label()
    print("\n✅ All tests passed!")
