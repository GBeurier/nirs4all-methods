//! Compare complete historical/current states across the writer-only ABI bump.
use n4m::{inspect_n4mm, roles::Estimator, Context};

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    end: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> &'a [u8] {
        let next = self.position.checked_add(count).unwrap();
        assert!(next <= self.end);
        let result = &self.bytes[self.position..next];
        self.position = next;
        result
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn count(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn text(&mut self) -> &'a str {
        let length = self.u32() as usize;
        std::str::from_utf8(self.take(length)).unwrap()
    }
}

fn nested_model(
    normalized: &mut [u8],
    packet: &[u8],
    start: usize,
    end: usize,
    abi: (u32, u32, u32),
) {
    assert!(end >= start + 28 && end <= packet.len() - 8);
    // Public native inspection checks the complete model and checksum. Every
    // byte outside the known writer fields remains in the final comparison.
    let info = inspect_n4mm(&packet[start..end]).unwrap();
    assert_eq!(info.writer_abi, abi);
    normalized[start + 8..start + 20].fill(0);
    normalized[end - 8..end].fill(0);
}

fn normalized(ctx: &Context, packet: &[u8], abi: (u32, u32, u32)) -> Vec<u8> {
    // Native import validates checksum, parameters and every learned block;
    // the test decoder below only locates the closed writer-field exceptions.
    let estimator = Estimator::from_n4me(ctx, packet).unwrap();
    assert!(packet.len() >= 28);
    let mut reader = Reader {
        bytes: packet,
        position: 0,
        end: packet.len() - 8,
    };
    assert_eq!(reader.take(4), b"N4ME");
    assert_eq!(reader.u32(), 1);
    assert_eq!((reader.u32(), reader.u32(), reader.u32()), abi);
    let method_id = reader.text();
    assert_eq!(estimator.method_id().unwrap(), method_id);
    for _ in 0..reader.u32() {
        reader.text();
        reader.u32();
        let count = reader.count().checked_mul(8).unwrap();
        reader.take(count);
    }
    reader.take(24); // capabilities, input features, outputs
    let mut result = packet.to_vec();
    result[8..20].fill(0);
    result[packet.len() - 8..].fill(0);
    for _ in 0..reader.u32() {
        let tag = reader.u32();
        let size = reader.count();
        let start = reader.position;
        let block = reader.take(size);
        if tag == 0x4d4d344e {
            nested_model(&mut result, packet, start, start + size, abi);
        } else if tag == 0x31534c43
            && matches!(
                method_id,
                "models.classification.pls_lda"
                    | "models.classification.pls_logistic"
                    | "models.sparse.sparse_pls_da"
            )
        {
            // Only these CLS1 adapters pack a length-framed N4MM. QDA and
            // every other adapter remain entirely byte-equal.
            let mut classifier = Reader {
                bytes: block,
                position: 8,
                end: block.len(),
            };
            let classes = classifier.count();
            assert!((2..=1 << 20).contains(&classes));
            classifier.take(classes.checked_mul(8).unwrap());
            let model_size = classifier.count();
            let packed_count = classifier.count();
            assert_eq!(packed_count, model_size.div_ceil(8));
            let model_start = start + classifier.position;
            classifier.take(packed_count.checked_mul(8).unwrap());
            nested_model(
                &mut result,
                packet,
                model_start,
                model_start + model_size,
                abi,
            );
        }
    }
    assert_eq!(reader.position, reader.end);
    result
}

pub fn assert_reexport_equivalent(ctx: &Context, original: &[u8], current: &[u8], method: &str) {
    let old = normalized(ctx, original, (2, 15, 0));
    let new = normalized(ctx, current, (2, 16, 0));
    assert!(
        old == new,
        "{method}: non-writer bytes changed during re-export"
    );
}
