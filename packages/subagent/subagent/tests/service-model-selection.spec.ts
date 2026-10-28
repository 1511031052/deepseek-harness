/**
 * The subagent service execution boundary enforces the shared model-selection
 * policy on explicit `request.agentOptions` before any provider creates a
 * child. These tests drive real `startActivation` calls, inspect the created
 * descriptor and adapter request, and prove rejected selections never reach
 * `ctx.agents.create`.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import type { Agent, AgentOptions } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type { LlmModelReasoningInfo, StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import * as SubagentSpawn from '@deepseek-ai/dsh-subagent-spawn-in-process'
import { MockAdapter, textResponse } from '../../../core/agent-loop/tests/mock-adapter.ts'
import SubagentRuntime, { foldSubagentDescriptor } from '@deepseek-ai/dsh-subagent'
import type { SubagentProvider } from '@deepseek-ai/dsh-subagent'
import { subagentModelSelectionPolicy } from '@deepseek-ai/dsh-subagent/model-selection'
import { mountWorkingDirectoryFixture } from './working-directory-fixture.ts'

const REASONING: LlmModelReasoningInfo = {
  efforts: [
    { id: ReasoningEffortId('high'), name: 'High' },
    { id: ReasoningEffortId('max'), name: 'Max' },
  ],
  defaultEffort: ReasoningEffortId('high'),
}

const ALLOWED = [{ provider: 'alpha', model: 'small' }] as const

const contexts: Context[] = []
let counter = 0

afterEach(async () => {
  for (const ctx of contexts.splice(0).reverse()) await ctx.fiber.dispose()
})

interface SetupOptions {
  readonly allowedModels?: readonly { provider: string; model: string }[] | null
  readonly parentOptions?: AgentOptions
  readonly script?: StreamChunk[]
}

/**
 * Mount the real service, spawn provider, AgentLoop, and one live parent.
 * A non-null `allowedModels` appends the durable policy before the call.
 */
async function setup(options: SetupOptions = {}) {
  const ctx = new Context()
  contexts.push(ctx)
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(AgentLoop, { agents: [] })
  await mountWorkingDirectoryFixture(ctx)
  await ctx.plugin(SubagentRuntime)
  await ctx.plugin(SubagentSpawn, { providerName: 'spawn' })
  const adapter = new MockAdapter(options.script ?? [textResponse('child done')], REASONING)
  ctx.llm.registerAdapter(['alpha', 'beta'], adapter)
  const parent = await ctx.agentLoop.create(
    SessionId(`parent-${++counter}`),
    options.parentOptions ?? { provider: 'alpha', model: 'parent' },
  )
  if (options.allowedModels !== undefined && options.allowedModels !== null) {
    parent.session.append('subagent/model-selection-policy', {
      allowedModels: options.allowedModels.map(route => ({ ...route })),
    })
  }
  return { ctx, parent, adapter }
}

/** Drive the service with an explicit selection rather than the model-facing tool. */
function start(ctx: Context, parent: Agent, agentOptions?: AgentOptions, provider = 'spawn') {
  return ctx.subagents.startActivation({
    provider,
    label: 'route test',
    request: {
      prompt: [{ type: 'text', text: 'do the routed task' }],
      parent,
      ...agentOptions === undefined ? {} : { agentOptions },
    },
    signal: new AbortController().signal,
    delivery: 'caller',
  })
}

/**
 * Register an external provider so route-omission tests observe only the
 * service boundary, never a driven child Agent.
 */
function registerExternalProvider(ctx: Context) {
  const startExternal = vi.fn(async () => ({
    id: SessionId(`external-${++counter}`),
    result: Promise.resolve({ output: [], stopReason: 'completed' as const }),
    dispose: () => Promise.resolve(),
  }))
  const provider: SubagentProvider = {
    name: 'external',
    capabilities: { agentOptions: true, agentPreset: false, outputSchema: true, depthLimit: false },
    inheritsParentContext: false,
    start: startExternal,
  }
  ctx.subagents.registerProvider(provider)
  return startExternal
}

describe('startActivation child LLM selection', () => {
  it('creates a policy-authorized route with its effort and records both', async () => {
    const { ctx, parent, adapter } = await setup({ allowedModels: ALLOWED })
    const started = await start(ctx, parent, {
      provider: 'alpha',
      model: 'small',
      reasoningEffort: ReasoningEffortId('max'),
    })
    try {
      const session = ctx.sessions.get(started.childId)
      expect(session).toBeDefined()
      expect(foldSubagentDescriptor(session!.snapshotEvents())).toMatchObject({
        mode: 'continuable',
        provider: 'spawn',
        agentProvider: 'alpha',
        agentModel: 'small',
        agentReasoningEffort: 'max',
      })
      await started.result
      expect(adapter.requests[0]).toMatchObject({
        provider: 'alpha',
        model: 'small',
        reasoningEffort: 'max',
      })
    } finally {
      await started.dispose()
    }
  })

  it('rejects a non-allowlisted explicit route before any child exists', async () => {
    const { ctx, parent } = await setup({ allowedModels: ALLOWED })
    const create = vi.spyOn(ctx.agents, 'create')
    const sessions = ctx.sessions.list().length

    await expect(start(ctx, parent, { provider: 'beta', model: 'wide' }))
      .rejects.toThrow('child LLM route "beta/wide" is not allowed for this Session')
    expect(create).not.toHaveBeenCalled()
    expect(ctx.sessions.list()).toHaveLength(sessions)
  })

  it('rejects an effort-only selection whose inherited parent route is not allowlisted', async () => {
    const { ctx, parent } = await setup({
      allowedModels: ALLOWED,
      parentOptions: { provider: 'alpha', model: 'other' },
    })
    const create = vi.spyOn(ctx.agents, 'create')

    await expect(start(ctx, parent, { reasoningEffort: ReasoningEffortId('max') }))
      .rejects.toThrow('child LLM route "alpha/other" is not allowed for this Session')
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects an effort the selected route does not advertise', async () => {
    const { ctx, parent } = await setup({ allowedModels: ALLOWED })
    const create = vi.spyOn(ctx.agents, 'create')

    await expect(start(ctx, parent, {
      provider: 'alpha',
      model: 'small',
      reasoningEffort: ReasoningEffortId('ultra'),
    })).rejects.toThrow('does not support reasoning effort "ultra"')
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects an explicit effort without an effective route', async () => {
    const { ctx, parent } = await setup({ allowedModels: ALLOWED, parentOptions: {} })
    const create = vi.spyOn(ctx.agents, 'create')

    await expect(start(ctx, parent, { reasoningEffort: ReasoningEffortId('max') }))
      .rejects.toThrow('cannot select child LLM values without an effective provider and model')
    expect(create).not.toHaveBeenCalled()
  })

  it('keeps the no-policy compatibility path but still preflights the live route', async () => {
    const { ctx, parent, adapter } = await setup({ allowedModels: null })
    const resolve = vi.spyOn(ctx.llm, 'resolveCallConfig')
    const started = await start(ctx, parent, { provider: 'alpha', model: 'small' })
    try {
      expect(resolve.mock.calls[0]?.[0]).toMatchObject({ provider: 'alpha', model: 'small' })
      const session = ctx.sessions.get(started.childId)
      expect(foldSubagentDescriptor(session!.snapshotEvents())).toMatchObject({
        agentProvider: 'alpha',
        agentModel: 'small',
      })
      await started.result
      expect(adapter.requests[0]).toMatchObject({ provider: 'alpha', model: 'small' })
    } finally {
      await started.dispose()
    }
  })

  it('rejects an invalid effort without a policy too', async () => {
    const { ctx, parent } = await setup({ allowedModels: null })
    const create = vi.spyOn(ctx.agents, 'create')

    await expect(start(ctx, parent, {
      provider: 'alpha',
      model: 'small',
      reasoningEffort: ReasoningEffortId('ultra'),
    })).rejects.toThrow('does not support reasoning effort "ultra"')
    expect(create).not.toHaveBeenCalled()
  })

  it.each([
    ['pure inheritance', undefined],
    ['a maxTokens-only override', { maxTokens: 1024 }],
  ] as const)('skips policy reads and preflight for %s', async (_label, agentOptions) => {
    const { ctx, parent } = await setup({ allowedModels: null })
    const startExternal = registerExternalProvider(ctx)
    const resolve = vi.spyOn(ctx.llm, 'resolveCallConfig')
    const stateOf = vi.spyOn(ctx.sessionProjections, 'stateOf')

    const started = await start(ctx, parent, agentOptions, 'external')
    try {
      expect(startExternal).toHaveBeenCalledOnce()
      expect(resolve).not.toHaveBeenCalled()
      expect(stateOf.mock.calls.filter(([, key]) => key === 'subagentModelSelectionPolicy')).toHaveLength(0)
    } finally {
      await started.dispose()
    }
  })
})

describe('subagent model-selection projection ownership', () => {
  it('registers the projection once when the service mounts', async () => {
    const ctx = new Context()
    contexts.push(ctx)
    await mountAgentLoopTestDependencies(ctx)
    await mountWorkingDirectoryFixture(ctx)
    const register = vi.spyOn(ctx.sessionProjections, 'register')

    await ctx.plugin(SubagentRuntime)

    const policyRegistrations = register.mock.calls.filter(([definition]) => definition.key === 'subagentModelSelectionPolicy')
    expect(policyRegistrations).toHaveLength(1)

    const session = ctx.sessions.create(SessionId(`projection-${++counter}`), {})
    expect(subagentModelSelectionPolicy(ctx.sessionProjections, session)).toBeUndefined()
    session.append('subagent/model-selection-policy', { allowedModels: [{ provider: 'alpha', model: 'small' }] })
    expect(subagentModelSelectionPolicy(ctx.sessionProjections, session)).toEqual([{ provider: 'alpha', model: 'small' }])
  })
})
