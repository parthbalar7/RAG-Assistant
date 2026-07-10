import { extractError } from '../api';

describe('extractError', () => {
  it('returns string detail', () => {
    expect(extractError({ detail: 'Not found' })).toBe('Not found');
  });

  it('returns array detail joined', () => {
    expect(extractError({ detail: [{ msg: 'field required' }, { msg: 'too short' }] })).toBe(
      'field required; too short',
    );
  });

  it('returns object detail message', () => {
    expect(extractError({ detail: { msg: 'bad input' } })).toBe('bad input');
  });

  it('returns message field', () => {
    expect(extractError({ message: 'Something went wrong' })).toBe('Something went wrong');
  });

  it('returns error field', () => {
    expect(extractError({ error: 'Server error' })).toBe('Server error');
  });

  it('returns fallback for null', () => {
    expect(extractError(null, 'fallback')).toBe('fallback');
  });

  it('returns default fallback for null without fallback', () => {
    expect(extractError(null)).toBe('Unknown error');
  });

  it('returns fallback for empty object', () => {
    expect(extractError({}, 'custom fallback')).toBe('custom fallback');
  });
});
