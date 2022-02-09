import hybrid
import dimod
import neal


class ComposedSampler:
    """
    Class for representing a composed sampler, which is able to break a large problem into subproblems and solve them
    in parallel with different samplers. The general structure is a composition of branches. These branches are
    composed as: branch = decomposer | sampler | composer.

    Possible decomposers are e.g.
    hybrid.EnergyImpactDecomposer: Selects variables according to their energy impact. Can be configured with e.g. BFS,
    hybrid.ComponentDecomposer: Selects a subproblem of variables that make up a connected component,
    hybrid.RandomSubproblemDecompose: Selects a subproblem of size random variables,
    but custom decomposers can also be implemented.
    """

    def __init__(self, sampler=None, decomposer=None, composer=None):
        self.workflow = None
        self.branches = []
        if sampler is not None:
            self.add_branch(sampler, decomposer, composer)

    def add_branch(self, sampler=None, decomposer=None, composer=None):
        if sampler is None:
            sampler = hybrid.SimulatedAnnealingProblemSampler()
        if not isinstance(sampler, hybrid.core.Runnable):
            raise TypeError("Sampler has to be of class hybrid.Runnable!")
        if isinstance(sampler, hybrid.traits.ProblemSampler) or isinstance(sampler, hybrid.Loop):
            branch = sampler
        elif isinstance(sampler, hybrid.traits.SubproblemSampler):
            if decomposer is None:
                decomposer = hybrid.IdentityDecomposer()
            else:
                self._check_decomposer(decomposer)
            if composer is None:
                composer = hybrid.SplatComposer()
            else:
                self._check_composer(composer)
            branch = decomposer | hybrid.Const(subsamples=None) | sampler | composer
            # decomposer | hybrid.Const(subsamples=None)
        else:
            raise TypeError("Sampler has to be either of class hybrid.traits.ProblemSampler,"
                            "hybrid.traits.SubproblemSampler or hybrid.Loop!")
        self.branches += [branch]
        return branch

    def add_parallel_subproblem_branch(self, sampler=None, decomposer=None, composer=None):
        """
        Method for processing many subproblems in parallel instead of sequentially adapting them.
        """

        if sampler is None:
            sampler = hybrid.SimulatedAnnealingSubproblemSampler()
        if not isinstance(sampler, hybrid.core.Runnable):
            raise TypeError("Sampler has to be of class hybrid.Runnable!")
        if decomposer is None:
            decomposer = hybrid.EnergyImpactDecomposer(size=2)
        else:
            self._check_decomposer(decomposer)
        if composer is None:
            composer = hybrid.SplatComposer()
        else:
            self._check_composer(composer)
        # Redefine the workflow: parallel subproblem solving for a single sampler
        subproblem = hybrid.Unwind(decomposer)

        # Helper function to merge subsamples in place
        def merge_substates(_, substates, **kwargs):
            a, b = substates
            return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))

        # Map sampler over all subproblems, then reduce subsamples by merging in place
        subsampler = hybrid.Map(sampler) | hybrid.Reduce(hybrid.Lambda(merge_substates)) | composer
        branch = subproblem | subsampler
        self.branches += [branch]
        return branch

    def update_workflow(self, racing=True, convergence=None, max_iter=None, **kwargs):
        if racing:
            iteration = hybrid.RacingBranches(*self.branches) | hybrid.ArgMin()
        else:
            iteration = hybrid.Branches(*self.branches) | hybrid.ArgMin()
        if convergence is None and max_iter is None:
            raise Exception("One of convergence and max_iter has to be set, infinite loop is created otherwise!")
        self.workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=convergence, max_iter=max_iter, **kwargs)

    @staticmethod
    def _check_decomposer(decomposer):
        if isinstance(decomposer, hybrid.core.Runnable):
            if isinstance(decomposer, hybrid.traits.ProblemDecomposer):
                pass
            else:
                raise TypeError("Decomposer has to be of class hybrid.traits.ProblemDecomposer!")
        else:
            raise TypeError("Decomposer has to be of class hybrid.Runnable!")

    @staticmethod
    def _check_composer(composer):
        if isinstance(composer, hybrid.core.Runnable):
            if isinstance(composer, hybrid.traits.SubsamplesComposer):
                pass
            else:
                raise TypeError("Composer has to be of class hybrid.traits.SubsamplesComposer!")
        else:
            raise TypeError("Composer has to be of class hybrid.Runnable!")

    def run(self):
        pass

    def hybrid_from_dimod(self, subproblem=True):
        pass


class ProblemSampler(hybrid.HybridRunnable, hybrid.traits.ProblemSampler):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` with `hybrid.traits.ProblemSampler`

    The runnable that samples from `state.problem` and populates `state.samples`.
    """

    def __init__(self, sampler, **sample_kwargs):
        super().__init__(
            sampler, fields=('problem', 'samples'), **sample_kwargs)


class SubproblemSampler(hybrid.HybridRunnable, hybrid.traits.SubproblemSampler):
    """Produces a `hybrid.Runnable` from a `dimod.Sampler` with `hybrid.traits.SubproblemSampler`

    The runnable that samples from `state.subproblem` and populates `state.subsamples`.
    """

    def __init__(self, sampler, **sample_kwargs):
        super().__init__(
            sampler, fields=('subproblem', 'subsamples'), **sample_kwargs)


if __name__ == "__main__":
    # Construct a problem
    bqm = dimod.BinaryQuadraticModel({'x': 0.0, 'y': 0.0, 'z': 8.0, 'a': 2.0, 'b': 0.0, 'c': 6.0},
                                     {('y', 'x'): 2.0, ('z', 'x'): -4.0, ('z', 'y'): -4.0,
                                      ('b', 'a'): 2.0, ('c', 'a'): -4.0, ('c', 'b'): -4.0, ('a', 'z'): -4.0},
                                     -1.0, 'BINARY')
    dictionary = {'shots': 2}
    sampler = neal.SimulatedAnnealingSampler()
    runnable = ProblemSampler(sampler, **dictionary)
    dictionary_sub = {'num_reads': 3}
    #print(isinstance(dictionary_sub['num_reads'], numbers.Integral))
    runnable_sub = SubproblemSampler(sampler, **dictionary_sub)
    runnable_qpu = hybrid.QPUSubproblemAutoEmbeddingSampler(qpu_sampler=sampler, **dictionary_sub)
    init_state = hybrid.State.from_problem(bqm)
    decomposer = hybrid.RandomSubproblemDecomposer(size=3)
    composed = ComposedSampler()
    # composed.add_branch(sampler=runnable_sub, decomposer=decomposer)
    composed.add_branch(sampler=runnable_qpu, decomposer=decomposer)
    #composed.add_branch(sampler=hybrid.SimulatedAnnealingSubproblemSampler(num_reads=3), decomposer=decomposer)
    composed.update_workflow(convergence=None, max_iter=10, racing=True)
    # iteration = hybrid.RacingBranches(*composed.branches) | hybrid.ArgMin()
    final_state = composed.workflow.run(init_state).result()
    print(final_state.samples.first)
