//! Static module image summary.

use crate::value::WasmVal;
use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use std::collections::BTreeMap;
use waffle::{entity::EntityRef, Func, Global, Memory, MemorySegment, Module, Table, WASM_PAGE};

#[derive(Clone, Debug)]
pub(crate) struct Image {
    pub memories: BTreeMap<Memory, MemImage>,
    pub globals: BTreeMap<Global, WasmVal>,
    pub tables: BTreeMap<Table, Vec<Func>>,
    pub stack_pointer: Option<Global>,
    pub main_heap: Option<Memory>,
    pub main_table: Option<Table>,
}

#[derive(Clone, Debug)]
pub(crate) struct MemImage {
    pub image: Vec<u8>,
}

impl MemImage {
    pub fn len(&self) -> usize {
        self.image.len()
    }
}

// To support modules that initialize their memory at instantiation time
// (including multi-threaded WASIX modules) we need to instantiate the module
// and then read the memory back instead of just relying on active segments.
pub(crate) fn build_image(
    module: &Module,
    store: &impl wizex::wasmer::AsStoreRef,
    instance: &wizex::wasmer::Instance,
    imported_memories: &[wizex::wasmer::Memory],
) -> anyhow::Result<Image> {
    let memories = imported_memories
        .iter()
        .chain(instance.exports.iter().filter_map(|e| {
            if let wizex::wasmer::Extern::Memory(memory) = e.1 {
                Some(memory)
            } else {
                None
            }
        }));
    Ok(Image {
        memories: memories
            .enumerate()
            .flat_map(|(id, mem)| maybe_mem_image(store, mem).map(|image| (Memory::new(id), image)))
            .collect(),
        globals: module
            .globals
            .entries()
            .flat_map(|(global_id, data)| match data.value {
                Some(bits) => Some((global_id, WasmVal::from_bits(data.ty, bits)?)),
                _ => None,
            })
            .collect(),
        tables: module
            .tables
            .entries()
            .map(|(id, data)| (id, data.func_elements.clone().unwrap_or(vec![])))
            .collect(),
        // HACK: assume first global is shadow stack pointer.
        stack_pointer: module.globals.iter().next(),
        // HACK: assume first memory is main heap.
        main_heap: module.memories.iter().next(),
        // HACK: assume first table is used for function pointers.
        main_table: module.tables.iter().next(),
    })
}

fn maybe_mem_image(
    store: &impl wizex::wasmer::AsStoreRef,
    mem: &wizex::wasmer::Memory,
) -> Option<MemImage> {
    Some(MemImage {
        image: mem
            .view(store)
            .copy_to_vec()
            .expect("should be able to read memory data"),
    })
}

pub(crate) fn update(module: &mut Module, im: &Image) {
    for (&mem_id, mem) in &im.memories {
        module.memories[mem_id].segments.clear();
        module.memories[mem_id].segments.push(MemorySegment {
            offset: 0,
            data: mem.image.clone(),
        });
        let image_pages = mem.image.len() / WASM_PAGE;
        module.memories[mem_id].initial_pages =
            std::cmp::max(module.memories[mem_id].initial_pages, image_pages);
    }
}

impl Image {
    pub(crate) fn can_read(&self, memory: Memory, addr: u32, size: u32) -> bool {
        let end = match addr.checked_add(size) {
            Some(end) => end,
            None => return false,
        };
        let image = match self.memories.get(&memory) {
            Some(image) => image,
            None => return false,
        };
        (end as usize) <= image.len()
    }

    pub(crate) fn main_heap(&self) -> anyhow::Result<Memory> {
        self.main_heap
            .ok_or_else(|| anyhow::anyhow!("no main heap"))
    }

    pub(crate) fn read_slice(&self, id: Memory, addr: u32, len: u32) -> anyhow::Result<&[u8]> {
        let image = self.memories.get(&id).unwrap();
        let addr = usize::try_from(addr).unwrap();
        let len = usize::try_from(len).unwrap();
        if addr + len >= image.len() {
            anyhow::bail!("Out of bounds");
        }
        Ok(&image.image[addr..(addr + len)])
    }

    pub(crate) fn read_u8(&self, id: Memory, addr: u32) -> anyhow::Result<u8> {
        let image = self.memories.get(&id).unwrap();
        image
            .image
            .get(addr as usize)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Out of bounds"))
    }

    pub(crate) fn read_u16(&self, id: Memory, addr: u32) -> anyhow::Result<u16> {
        let image = self.memories.get(&id).unwrap();
        let addr = addr as usize;
        if (addr + 2) > image.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &image.image[addr..(addr + 2)];
        Ok(u16::from_le_bytes([slice[0], slice[1]]))
    }

    pub(crate) fn read_u32(&self, id: Memory, addr: u32) -> anyhow::Result<u32> {
        let image = self.memories.get(&id).unwrap();
        let addr = addr as usize;
        if (addr + 4) > image.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &image.image[addr..(addr + 4)];
        Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    pub(crate) fn read_u64(&self, id: Memory, addr: u32) -> anyhow::Result<u64> {
        let low = self.read_u32(id, addr)?;
        let high = self.read_u32(id, addr + 4)?;
        Ok((high as u64) << 32 | (low as u64))
    }

    pub(crate) fn read_u128(&self, id: Memory, addr: u32) -> anyhow::Result<u128> {
        let low = self.read_u64(id, addr)?;
        let high = self.read_u64(id, addr + 8)?;
        Ok((high as u128) << 64 | (low as u128))
    }

    pub(crate) fn read_size(&self, id: Memory, addr: u32, size: u8) -> anyhow::Result<u64> {
        match size {
            1 => self.read_u8(id, addr).map(|x| x as u64),
            2 => self.read_u16(id, addr).map(|x| x as u64),
            4 => self.read_u32(id, addr).map(|x| x as u64),
            8 => self.read_u64(id, addr),
            _ => panic!("bad size"),
        }
    }

    pub(crate) fn read_str(&self, id: Memory, mut addr: u32) -> anyhow::Result<String> {
        let mut bytes = vec![];
        loop {
            let byte = self.read_u8(id, addr)?;
            if byte == 0 {
                break;
            }
            bytes.push(byte);
            addr += 1;
        }
        Ok(std::str::from_utf8(&bytes[..])?.to_owned())
    }

    pub(crate) fn write_u8(&mut self, id: Memory, addr: u32, value: u8) -> anyhow::Result<()> {
        let image = self.memories.get_mut(&id).unwrap();
        *image
            .image
            .get_mut(addr as usize)
            .ok_or_else(|| anyhow::anyhow!("Out of bounds"))? = value;
        Ok(())
    }

    pub(crate) fn write_u32(&mut self, id: Memory, addr: u32, value: u32) -> anyhow::Result<()> {
        let image = self.memories.get_mut(&id).unwrap();
        let addr = addr as usize;
        if (addr + 4) > image.len() {
            anyhow::bail!("Out of bounds");
        }
        let slice = &mut image.image[addr..(addr + 4)];
        slice.copy_from_slice(&value.to_le_bytes()[..]);
        Ok(())
    }

    pub(crate) fn func_ptr(&self, idx: u32) -> anyhow::Result<Func> {
        let table = self
            .main_table
            .ok_or_else(|| anyhow::anyhow!("no main table"))?;
        Ok(self
            .tables
            .get(&table)
            .unwrap()
            .get(idx as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("func ptr out of bounds"))?)
    }

    pub(crate) fn append_data(&mut self, id: Memory, data: Vec<u8>) {
        let image = self.memories.get_mut(&id).unwrap();
        let orig_len = image.len();
        let data_len = data.len();
        let padded_len = (data_len + WASM_PAGE - 1) & !(WASM_PAGE - 1);
        let padding = padded_len - data_len;
        image
            .image
            .extend(data.into_iter().chain(std::iter::repeat(0).take(padding)));
        log::debug!(
            "Appending data ({} bytes, {} padding): went from {} bytes to {} bytes",
            data_len,
            padding,
            orig_len,
            image.len()
        );
    }

    pub(crate) fn to_wizex_snapshot(&self) -> wizex::Snapshot {
        let memory = self.main_heap().expect("module should have a memory");

        let (segments, pages) = memory_data_to_snapshot_segments(
            memory
                .maybe_index()
                .expect("main memory should have valid index") as u32,
            &self.memories.get(&memory).unwrap().image,
        );

        wizex::Snapshot {
            // HACK: leave the globals empty to transcrive the globals section.
            globals: vec![],
            data_segments: segments,
            memory_mins: vec![pages],
        }
    }
}

// TODO: copy-pasted from wizex with minimal changes. should clean this up.
fn memory_data_to_snapshot_segments(
    memory_index: u32,
    memory_data: &[u8],
) -> (Vec<wizex::DataSegment>, u32) {
    let mut data_segments = vec![];
    let num_wasm_pages = memory_data.len() / WASM_PAGE;

    // Consider each Wasm page in parallel. Create data segments for each
    // region of non-zero memory.
    data_segments.par_extend((0..num_wasm_pages).into_par_iter().flat_map(|i| {
        let page_end = (i + 1) * WASM_PAGE;
        let mut start = i * WASM_PAGE;
        let mut segments = vec![];
        while start < page_end {
            let nonzero = match memory_data[start..page_end]
                .iter()
                .position(|byte| *byte != 0)
            {
                None => break,
                Some(i) => i,
            };
            start += nonzero;
            let end = memory_data[start..page_end]
                .iter()
                .position(|byte| *byte == 0)
                .map_or(page_end, |zero| start + zero);
            segments.push(wizex::DataSegment {
                memory_index,
                offset: u32::try_from(start).unwrap(),
                data: memory_data[start..end].to_vec(),
            });
            start = end;
        }
        segments
    }));

    if data_segments.is_empty() {
        return (data_segments, num_wasm_pages as u32);
    }

    // Sort data segments to enforce determinism in the face of the
    // parallelism above.
    data_segments.sort_by_key(|s| (s.memory_index, s.offset));

    // Merge any contiguous segments (caused by spanning a Wasm page boundary,
    // and therefore created in separate logical threads above) or pages that
    // are within four bytes of each other. Four because this is the minimum
    // overhead of defining a new active data segment: one for the memory index
    // LEB, two for the memory offset init expression (one for the `i32.const`
    // opcode and another for the constant immediate LEB), and finally one for
    // the data length LEB).
    const MIN_ACTIVE_SEGMENT_OVERHEAD: u32 = 4;
    let mut merged_data_segments = Vec::with_capacity(data_segments.len());
    merged_data_segments.push(data_segments[0].clone());
    for b in &data_segments[1..] {
        let a = merged_data_segments.last_mut().unwrap();

        // Only merge segments for the same memory.
        if a.memory_index != b.memory_index {
            merged_data_segments.push(b.clone());
            continue;
        }

        // Only merge segments if they are contiguous or if it is definitely
        // more size efficient than leaving them apart.
        let gap = a.gap(b);
        if gap > MIN_ACTIVE_SEGMENT_OVERHEAD {
            merged_data_segments.push(b.clone());
            continue;
        }

        // Okay, merge them together into `a` (so that the next iteration can
        // merge it with its predecessor) and then omit `b`!
        a.merge(b);
    }

    remove_excess_segments(&mut merged_data_segments);

    for s in merged_data_segments.iter().take(100) {
        println!("{}-{}", s.offset, s.len());
    }

    (merged_data_segments, num_wasm_pages as u32)
}

/// The maximum number of data segments that we will emit. Most
/// engines support more than this, but we want to leave some
/// headroom.
const MAX_DATA_SEGMENTS: usize = 10_000;

/// Engines apply a limit on how many segments a module may contain, and Wizer
/// can run afoul of it. When that happens, we need to merge data segments
/// together until our number of data segments fits within the limit.
fn remove_excess_segments(merged_data_segments: &mut Vec<wizex::DataSegment>) {
    if merged_data_segments.len() < MAX_DATA_SEGMENTS {
        return;
    }

    // We need to remove `excess` number of data segments.
    let excess = merged_data_segments.len() - MAX_DATA_SEGMENTS;

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct GapIndex {
        gap: u32,
        // Use a `u32` instead of `usize` to fit `GapIndex` within a word on
        // 64-bit systems, using less memory.
        index: u32,
    }

    // Find the gaps between the start of one segment and the next (if they are
    // both in the same memory). We will merge the `excess` segments with the
    // smallest gaps together. Because they are the smallest gaps, this will
    // bloat the size of our data segment the least.
    let mut smallest_gaps = Vec::with_capacity(merged_data_segments.len() - 1);
    for (index, w) in merged_data_segments.windows(2).enumerate() {
        if w[0].memory_index != w[1].memory_index {
            continue;
        }
        let gap = w[0].gap(&w[1]);
        let index = u32::try_from(index).unwrap();
        smallest_gaps.push(GapIndex { gap, index });
    }
    smallest_gaps.sort_unstable_by_key(|g| g.gap);
    smallest_gaps.truncate(excess);

    // Now merge the chosen segments together in reverse index order so that
    // merging two segments doesn't mess up the index of the next segments we
    // will to merge.
    smallest_gaps.sort_unstable_by(|a, b| a.index.cmp(&b.index).reverse());
    for GapIndex { index, .. } in smallest_gaps {
        let index = usize::try_from(index).unwrap();

        // array[i].do_something(array[j]) is a reborrow, so we split the array
        // into two slices that we can borrow from simultaneously.
        let (first, second) = &mut merged_data_segments[..].split_at_mut(index + 1);
        first[index].merge(&second[0]);

        // Okay to use `swap_remove` here because, even though it makes
        // `merged_data_segments` unsorted, the segments are still sorted within
        // the range `0..index` and future iterations will only operate within
        // that subregion because we are iterating over largest to smallest
        // indices.
        merged_data_segments.swap_remove(index + 1);
    }

    // Finally, sort the data segments again so that our output is
    // deterministic.
    merged_data_segments.sort_by_key(|s| (s.memory_index, s.offset));
}
