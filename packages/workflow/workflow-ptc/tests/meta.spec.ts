import { describe, expect, it } from 'vitest'
import { WorkflowError } from '@deepseek-ai/dsh-workflow'
import { validateMeta } from '../src/meta.ts'

/** Assert a META_INVALID throw whose message matches every given fragment. */
function expectInvalid(value: unknown, ...fragments: string[]): void {
  let thrown: unknown
  try {
    validateMeta(value)
  } catch (error: unknown) {
    thrown = error
  }
  expect(thrown).toBeInstanceOf(WorkflowError)
  expect((thrown as WorkflowError).code).toBe('META_INVALID')
  for (const fragment of fragments) {
    expect((thrown as WorkflowError).message).toContain(fragment)
  }
}

describe('validateMeta', () => {
  it('accepts a minimal meta and returns a normalized copy (no aliasing of the input)', () => {
    const input = { name: 'audit', description: 'audit the repo' }
    const meta = validateMeta(input)
    expect(meta).toEqual({ name: 'audit', description: 'audit the repo' })
    expect(meta).not.toBe(input)
    input.name = 'mutated'
    expect(meta.name).toBe('audit')
  })

  it('accepts the full shape and rebuilds phases entry by entry', () => {
    const meta = validateMeta({
      name: 'migrate',
      description: 'migrate call sites',
      whenToUse: 'large mechanical sweeps',
      phases: [
        { title: 'Discover', provider: 'openai', model: 'gpt-small' },
        { title: 'Transform', detail: 'one agent per file', provider: 'deepseek', model: 'deepseek-v4-pro', reasoning_effort: 'max' },
      ],
    })
    expect(meta).toEqual({
      name: 'migrate',
      description: 'migrate call sites',
      whenToUse: 'large mechanical sweeps',
      phases: [
        { title: 'Discover', provider: 'openai', model: 'gpt-small' },
        { title: 'Transform', detail: 'one agent per file', provider: 'deepseek', model: 'deepseek-v4-pro', reasoning_effort: 'max' },
      ],
    })
  })

  it('rejects non-object values loud', () => {
    expectInvalid(undefined, 'meta must be an object')
    expectInvalid('a string', 'meta must be an object')
    expectInvalid(null, 'meta must be an object')
    expectInvalid([{ name: 'x', description: 'd' }], 'meta must be an object')
  })

  it('rejects unknown fields by name (accepted-then-ignored is banned)', () => {
    expectInvalid({ name: 'x', description: 'd', color: 'red' }, 'meta.color is not a recognized field')
  })

  it('rejects missing or mistyped name/description/whenToUse', () => {
    expectInvalid({ description: 'd' }, 'meta.name must be a non-empty string')
    expectInvalid({ name: '', description: 'd' }, 'meta.name must be a non-empty string')
    expectInvalid({ name: 'x' }, 'meta.description must be a non-empty string')
    expectInvalid({ name: 'x', description: 42 }, 'meta.description must be a non-empty string')
    expectInvalid({ name: 'x', description: 'd', whenToUse: 3 }, 'meta.whenToUse must be a string')
  })

  it('rejects malformed phases, entry by entry', () => {
    expectInvalid({ name: 'x', description: 'd', phases: 'Scan' }, 'meta.phases must be an array')
    expectInvalid({ name: 'x', description: 'd', phases: ['Scan'] }, 'meta.phases[0] must be an object')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: '' }] }, 'meta.phases[0].title must be a non-empty string')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', order: 1 }] }, 'meta.phases[0].order is not a recognized field')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', detail: 9 }] }, 'meta.phases[0].detail must be a string')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', provider: 9 }] }, 'meta.phases[0].provider must be a string')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', model: 9 }] }, 'meta.phases[0].model must be a string')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', reasoning_effort: 9 }] }, 'meta.phases[0].reasoning_effort must be a string')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', provider: '' }] }, 'meta.phases[0].provider must be non-empty')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', model: '' }] }, 'meta.phases[0].model must be non-empty')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', reasoning_effort: '' }] }, 'meta.phases[0].reasoning_effort must be non-empty')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', provider: 'openai' }] }, 'meta.phases[0] must supply provider and model together')
    expectInvalid({ name: 'x', description: 'd', phases: [{ title: 'Scan', model: 'small' }] }, 'meta.phases[0] must supply provider and model together')
  })

  it('keeps an effort without a route legal because it applies to the inherited route', () => {
    const meta = validateMeta({
      name: 'review',
      description: 'adversarial review',
      phases: [{ title: 'Check', reasoning_effort: 'max' }],
    })
    expect(meta.phases).toEqual([{ title: 'Check', reasoning_effort: 'max' }])
  })

  it('names EVERY violation in one throw, not just the first', () => {
    expectInvalid(
      { description: 7, extra: true, phases: [{ title: 'Scan' }, 'bad'] },
      'meta.extra is not a recognized field',
      'meta.name must be a non-empty string',
      'meta.description must be a non-empty string',
      'meta.phases[1] must be an object',
    )
  })
})
