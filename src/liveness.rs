//! Liveness analysis: analyze the register pressure of original and
//! specialized functions.

use fxhash::FxHashSet;
use std::collections::VecDeque;
use waffle::{cfg::CFGInfo, entity::PerEntity, Block, FunctionBody, Terminator, Value, ValueDef};

pub(crate) type LiveSet = FxHashSet<Value>;

#[derive(Clone, Debug)]
pub(crate) struct Liveness<'a> {
    pub func: &'a FunctionBody,
    pub block_start: PerEntity<Block, LiveSet>,
    pub block_end: PerEntity<Block, LiveSet>,
}

pub(crate) fn scan_block_backward<T, Use: Fn(&mut T, Value), Def: Fn(&mut T, Value)>(
    func: &FunctionBody,
    block: Block,
    state: &mut T,
    use_func: Use,
    def_func: Def,
) {
    func.blocks[block].terminator.visit_uses(|arg| {
        let arg = func.resolve_alias(arg);
        use_func(state, arg);
    });

    for &value in func.blocks[block].insts.iter().rev() {
        match &func.values[value] {
            &ValueDef::Operator(_op, args, _) => {
                def_func(state, value);
                for &arg in &func.arg_pool[args] {
                    let arg = func.resolve_alias(arg);
                    use_func(state, arg);
                }
            }
            &ValueDef::PickOutput(arg, _, _) => {
                let arg = func.resolve_alias(arg);
                def_func(state, value);
                use_func(state, arg);
            }
            &ValueDef::Alias(_) => {}
            &ValueDef::BlockParam(..) | &ValueDef::Placeholder(_) | &ValueDef::None => {
                unreachable!()
            }
        }
    }

    for &(_, arg) in &func.blocks[block].params {
        def_func(state, arg);
    }
}

impl<'a> Liveness<'a> {
    pub(crate) fn new(func: &'a FunctionBody, cfg: &CFGInfo) -> Liveness<'a> {
        let mut this = Liveness {
            func,
            block_start: PerEntity::default(),
            block_end: PerEntity::default(),
        };

        let mut workqueue = VecDeque::new();
        let mut workqueue_set = FxHashSet::default();
        let mut processed = FxHashSet::default();
        for (block, block_def) in func.blocks.entries() {
            match &block_def.terminator {
                Terminator::Return { .. } | Terminator::Unreachable => {
                    workqueue.push_back(block);
                    workqueue_set.insert(block);
                    processed.insert(block);
                }
                _ => {}
            }
        }

        while let Some(block) = workqueue.pop_front() {
            workqueue_set.remove(&block);
            let mut liveness = this.block_end[block].clone();
            scan_block_backward(
                func,
                block,
                &mut liveness,
                |liveness, use_value| {
                    liveness.insert(use_value);
                },
                |liveness, def_value| {
                    liveness.remove(&def_value);
                },
            );
            this.block_start[block] = liveness.clone();

            for &pred in &cfg.preds[block] {
                let mut changed = processed.insert(pred);
                let pred_live = &mut this.block_end[pred];
                for &value in &liveness {
                    if pred_live.insert(value) {
                        changed = true;
                    }
                }
                if changed {
                    if workqueue_set.insert(pred) {
                        workqueue.push_back(pred);
                    }
                }
            }
        }

        this
    }
}
