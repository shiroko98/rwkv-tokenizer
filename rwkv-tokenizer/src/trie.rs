#[derive(Default, Debug)]
struct TrieNode {
    children: [[Option<Box<TrieNode>>; 16]; 16],
    id: Option<u16>,
}

impl TrieNode {
    fn new() -> Self {
        let mut trinode = TrieNode {
            children: Default::default(),
            id: None,
        };
        for index in 0..256 {
            trinode.children[index >> 4][index & 15] = None;
        }
        trinode
    }
}

#[derive(Debug)]
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    pub(crate) fn new() -> Self {
        Trie {
            root: TrieNode::new(),
        }
    }

    pub(crate) fn insert(&mut self, word: &Vec<u8>, id: u16) {
        let mut node = &mut self.root;
        for ch in word {
            let ch = u8::from_be(*ch) as usize;
            let index_a = ch >> 4;
            let index_b = ch & 15;
            if node.children[index_a][index_b].is_none() {
                node.children[index_a][index_b] = Option::from(Box::new(TrieNode::new()));
            }
            match &mut node.children[index_a][index_b] {
                Some(next_node) => node = next_node,
                None => unreachable!(), // We've just checked that it's not None
            }
        }
        node.id = Some(id)
    }

    fn search_the_longest(&self, word: &[u8]) -> Option<(usize, u16)> {
        let mut node = &self.root;
        let mut best: Option<(usize, u16)> = None;
        for (index, ch) in word.iter().enumerate() {
            let ch = u8::from_be(*ch) as usize;
            let index_a = ch >> 4;
            let index_b = ch & 15;
            if let Some(next_node) = &node.children[index_a][index_b] {
                node = &next_node;
                if let Some(id) = node.id {
                    best = Some((index + 1, id));
                }
            } else {
                return best;
            }
        }
        best
    }

    pub(crate) fn tokenize(&self, text: &str) -> Vec<u16> {
        let mut vec: Vec<u16> = Vec::new();
        let text_length = text.len();
        let mut index: usize = 0;
        while index < text_length {
            match self.search_the_longest(&text.as_bytes()[index..]) {
                Some((token_len, id)) => {
                    vec.push(id.into());
                    index += token_len;
                }
                None => return vec,
            }
        }
        vec
    }
}
