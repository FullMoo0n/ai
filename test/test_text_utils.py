"""Unit tests for text_utils module"""

import sys
sys.path.insert(0, '/home/runner/work/ai/ai')

from app.utils.text_utils import clean_markdown_json_response


def test_basic_json_markdown():
    """Test basic JSON markdown block removal"""
    input_text = '```json\n{"key": "value"}\n```'
    expected = '{"key": "value"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Basic JSON markdown test passed")


def test_basic_markdown():
    """Test basic markdown block removal without json tag"""
    input_text = '```\n{"key": "value"}\n```'
    expected = '{"key": "value"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Basic markdown test passed")


def test_no_markdown():
    """Test input without markdown blocks"""
    input_text = '{"key": "value"}'
    expected = '{"key": "value"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ No markdown test passed")


def test_whitespace_handling():
    """Test proper whitespace handling"""
    input_text = '  ```json  \n{"a": 1}  \n```  '
    expected = '{"a": 1}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Whitespace handling test passed")


def test_empty_string():
    """Test empty string input"""
    input_text = ''
    expected = ''
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Empty string test passed")


def test_whitespace_only():
    """Test string with only whitespace"""
    input_text = '   \n  \t  '
    expected = ''
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Whitespace only test passed")


def test_markdown_with_newlines():
    """Test markdown block with multiple newlines"""
    input_text = '```json\n\n{"key": "value"}\n\n```'
    expected = '{"key": "value"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Markdown with newlines test passed")


def test_only_opening_markdown():
    """Test markdown with only opening block"""
    input_text = '```json\n{"key": "value"}'
    expected = '{"key": "value"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Only opening markdown test passed")


def test_only_closing_markdown():
    """Test markdown with only closing block"""
    input_text = '{"key": "value"}\n```'
    expected = '{"key": "value"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Only closing markdown test passed")


def test_complex_json():
    """Test with complex nested JSON"""
    input_text = '''```json
{
  "sentences": {
    "한글": ["형태소1", "형태소2"],
    "nested": {
      "key": "value"
    }
  }
}
```'''
    result = clean_markdown_json_response(input_text)
    assert result.startswith('{') and result.endswith('}')
    assert '```' not in result
    print("✅ Complex JSON test passed")


def test_markdown_in_middle():
    """Test that markdown in middle is not removed (only start/end)"""
    input_text = '{"code": "```example```"}'
    expected = '{"code": "```example```"}'
    result = clean_markdown_json_response(input_text)
    assert result == expected, f"Expected {repr(expected)}, got {repr(result)}"
    print("✅ Markdown in middle test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running text_utils unit tests")
    print("=" * 60)
    
    tests = [
        test_basic_json_markdown,
        test_basic_markdown,
        test_no_markdown,
        test_whitespace_handling,
        test_empty_string,
        test_whitespace_only,
        test_markdown_with_newlines,
        test_only_opening_markdown,
        test_only_closing_markdown,
        test_complex_json,
        test_markdown_in_middle,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} raised exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"✅ All {len(tests)} tests passed!")
    else:
        print(f"❌ {failed}/{len(tests)} tests failed!")
    print("=" * 60)
    
    sys.exit(failed)
